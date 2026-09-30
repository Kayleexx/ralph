//! SQLite-backed session metadata store.
//!
//! Every state-changing write goes through one `rusqlite` transaction, relying on
//! SQLite's own WAL guarantees for crash-safety instead of a hand-rolled
//! temp-file+fsync+rename scheme.
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;

use rusqlite::{Connection, OptionalExtension, params};
use thiserror::Error;

use crate::continuity::ContinuityPolicy;
use crate::state::SessionState;
pub mod checkpoints;
pub mod handoff;
mod migrate;
mod profiles;
mod resolve;
mod restarts;
#[cfg(test)]
mod tests;
pub(crate) mod token_log;
pub(crate) mod turn_export;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("session name {0:?} already exists")]
    DuplicateName(String),
    #[error("no session found matching {identifier:?}")]
    NotFound {
        identifier: String,
        suggestion: Option<String>,
    },
    #[error("{0:?} matches more than one session: {1:?}")]
    AmbiguousPrefix(String, Vec<String>),
    #[error("storage is temporarily busy, try again")]
    Busy,
    #[error("persistent state is corrupt: {0}")]
    Corrupt(String),
    #[error(transparent)]
    Sqlite(rusqlite::Error),
}

impl From<rusqlite::Error> for StorageError {
    fn from(e: rusqlite::Error) -> Self {
        if matches!(
            e,
            rusqlite::Error::FromSqlConversionFailure(..) | rusqlite::Error::InvalidColumnType(..)
        ) {
            return StorageError::Corrupt(e.to_string());
        }
        if let rusqlite::Error::SqliteFailure(inner, _) = &e {
            match inner.code {
                rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked => {
                    return StorageError::Busy;
                }
                rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase => {
                    return StorageError::Corrupt(e.to_string());
                }
                _ => {}
            }
        }
        StorageError::Sqlite(e)
    }
}

#[derive(Debug, Clone)]
pub struct SessionRow {
    pub id: String,
    pub name: String,
    pub model: String,
    /// `None` until resolved (or if it never could be) — never a fake value equal to
    /// `model` itself.
    pub model_revision: Option<String>,
    pub tokenizer_revision: Option<String>,
    pub engine: String,
    pub engine_version: Option<String>,
    pub state: SessionState,
    pub pid: Option<i64>,
    pub location: String,
    pub token_count: i64,
    pub created_at: String,
    pub updated_at: String,
    pub continuity_policy: ContinuityPolicy,
    /// `ralph run --continuity-target <ms>` — a hint, not a guarantee (RALPH continuity
    /// phase 4): how fast this session's owner wants it restorable if demoted.
    pub continuity_target_ms: Option<i64>,
}

pub struct Storage {
    conn: Connection,
}

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS sessions (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    model               TEXT NOT NULL,
    model_revision      TEXT,
    tokenizer_revision  TEXT,
    engine              TEXT NOT NULL,
    engine_version      TEXT,
    state               TEXT NOT NULL,
    pid                 INTEGER,
    location            TEXT NOT NULL DEFAULT 'local/gpu0',
    token_count         INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    continuity_policy   TEXT NOT NULL DEFAULT 'warm',
    continuity_target_ms INTEGER
)";

impl Storage {
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(path)
            .map_err(|e| StorageError::Corrupt(format!("cannot open durable state: {e}")))?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| StorageError::Corrupt(format!("cannot protect durable state: {e}")))?;
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        let check: String = conn.pragma_query_value(None, "quick_check", |row| row.get(0))?;
        if check != "ok" {
            return Err(StorageError::Corrupt(check));
        }
        // Must run before the `CREATE TABLE IF NOT EXISTS` below: for an existing
        // database on an older schema, this brings it up to date; for a brand-new one,
        // it's a no-op and the create below does the real work.
        migrate::run(&conn)?;
        conn.execute(SCHEMA, [])?;
        Ok(Self { conn })
    }

    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self, StorageError> {
        let conn = Connection::open_in_memory()?;
        migrate::run(&conn)?;
        conn.execute(SCHEMA, [])?;
        Ok(Self { conn })
    }

    #[cfg(test)]
    pub(crate) fn test_execute(&self, sql: &str) {
        self.conn.execute_batch(sql).unwrap();
    }

    pub fn name_taken(&self, name: &str) -> Result<bool, StorageError> {
        let exists: Option<i64> = self
            .conn
            .query_row(
                "SELECT 1 FROM sessions WHERE name = ?1",
                params![name],
                |r| r.get(0),
            )
            .optional()?;
        Ok(exists.is_some())
    }

    pub fn insert(&self, row: &SessionRow) -> Result<(), StorageError> {
        let tx = self.conn.unchecked_transaction()?;
        let inserted = tx.execute(
            "INSERT INTO sessions (
                id, name, model, model_revision, tokenizer_revision, engine, engine_version,
                state, pid, location, token_count, created_at, updated_at,
                continuity_policy, continuity_target_ms
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                row.id,
                row.name,
                row.model,
                row.model_revision,
                row.tokenizer_revision,
                row.engine,
                row.engine_version,
                row.state.to_string(),
                row.pid,
                row.location,
                row.token_count,
                row.created_at,
                row.updated_at,
                row.continuity_policy.to_string(),
                row.continuity_target_ms,
            ],
        );
        match inserted {
            Ok(_) => {
                tx.commit()?;
                Ok(())
            }
            // Only a UNIQUE violation on `name` specifically is ever a duplicate name.
            // Any other constraint failure (NOT NULL, a different UNIQUE index, ...) is
            // a real, distinct bug and must never be misreported as one — that exact
            // conflation is what silently turned a schema mismatch into a confusing
            // "session already exists" for a name that had never been used.
            Err(rusqlite::Error::SqliteFailure(e, Some(ref msg)))
                if e.code == rusqlite::ErrorCode::ConstraintViolation
                    && msg.contains("sessions.name") =>
            {
                Err(StorageError::DuplicateName(row.name.clone()))
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Atomically updates state, pid, and updated_at together — the minimal set of fields
    /// that must move in lockstep so a crash mid-write never leaves an inconsistent row.
    pub fn set_state(
        &self,
        id: &str,
        state: SessionState,
        pid: Option<i64>,
        updated_at: &str,
    ) -> Result<(), StorageError> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "UPDATE sessions SET state = ?1, pid = ?2, updated_at = ?3 WHERE id = ?4",
            params![state.to_string(), pid, updated_at, id],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn set_resolved(
        &self,
        id: &str,
        model_revision: Option<&str>,
        engine_version: Option<&str>,
        updated_at: &str,
    ) -> Result<(), StorageError> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "UPDATE sessions SET model_revision = ?1, tokenizer_revision = ?1, engine_version = ?2, updated_at = ?3 WHERE id = ?4",
            params![model_revision, engine_version, updated_at, id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Written only after a migration's destination worker is fully attached and primed
    /// — the commit point `daemon::migrate` relies on for crash-safe rollback.
    pub fn set_location(
        &self,
        id: &str,
        location: &str,
        updated_at: &str,
    ) -> Result<(), StorageError> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "UPDATE sessions SET location = ?1, updated_at = ?2 WHERE id = ?3",
            params![location, updated_at, id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Bumps only `updated_at`, so per-session idle tracking (`daemon::idle`) reflects
    /// real query activity, not just state transitions.
    pub fn touch(&self, id: &str, updated_at: &str) -> Result<(), StorageError> {
        self.conn.execute(
            "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
            params![updated_at, id],
        )?;
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<SessionRow>, StorageError> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM sessions ORDER BY created_at")?;
        let rows = stmt.query_map([], row_to_session)?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::from)
    }

    /// Resolves a session by exact name, exact ID, or a unique ID prefix — never a fuzzy
    /// match.
    /// Any session left in `Starting`, or `Active` with no currently-live worker, is
    /// demoted to `to` (never left pretending to be usable) — called on daemon startup to
    /// reconcile state after a crash or reboot.
    pub fn reconcile_after_restart(
        &self,
        is_pid_alive: impl Fn(i64) -> bool,
        to: SessionState,
        now: &str,
    ) -> Result<Vec<String>, StorageError> {
        let mut demoted = Vec::new();
        for row in self.list()? {
            let stale = match row.state {
                SessionState::Starting | SessionState::Recovering => true,
                SessionState::Active => match row.pid {
                    Some(pid) => !is_pid_alive(pid),
                    None => true,
                },
                _ => false,
            };
            if stale {
                self.set_state(&row.id, to, None, now)?;
                demoted.push(row.id);
            }
        }
        Ok(demoted)
    }

    /// A daemon crash mid-operation can only ever leave a session in one of these
    /// transient states — each is the very last thing to flip before a stable landing
    /// state, and by the time this runs, `ownership::reconcile` has already killed any
    /// worker process the previous daemon still had running, so which stable state we
    /// land in is always safe regardless of exactly how far the interrupted operation
    /// got (see `state.rs`'s in-transit-state tests for the completeness proof):
    /// `Moving` (handoff/drain never actually left this machine until its final commit)
    /// and `Resuming` both land in `Paused`; `Pausing`/`Hibernating` land in the state
    /// they were already about to reach. This daemon process starts with no in-memory
    /// worker handles either way, so nothing here is ever silently `Active` again —
    /// `ralph resume` picks fast vs. portable from the landing state exactly like any
    /// other paused/hibernated session.
    pub fn rollback_stuck_transitions(&self, now: &str) -> Result<Vec<String>, StorageError> {
        let mut rolled_back = Vec::new();
        for row in self.list()? {
            let landing = match row.state {
                SessionState::Moving | SessionState::Resuming | SessionState::Pausing => {
                    Some(SessionState::Paused)
                }
                SessionState::Hibernating => Some(SessionState::Hibernated),
                _ => None,
            };
            if let Some(landing) = landing {
                self.set_state(&row.id, landing, None, now)?;
                rolled_back.push(row.id);
            }
        }
        Ok(rolled_back)
    }
}

pub(super) fn row_to_session(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionRow> {
    let state_str: String = row.get("state")?;
    let state = state_str.parse::<SessionState>().map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let policy_str: String = row.get("continuity_policy")?;
    let continuity_policy = policy_str.parse::<ContinuityPolicy>().map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    Ok(SessionRow {
        id: row.get("id")?,
        name: row.get("name")?,
        model: row.get("model")?,
        model_revision: row.get("model_revision")?,
        tokenizer_revision: row.get("tokenizer_revision")?,
        engine: row.get("engine")?,
        engine_version: row.get("engine_version")?,
        state,
        pid: row.get("pid")?,
        location: row.get("location")?,
        token_count: row.get("token_count")?,
        created_at: row.get("created_at")?,
        updated_at: row.get("updated_at")?,
        continuity_policy,
        continuity_target_ms: row.get("continuity_target_ms")?,
    })
}
