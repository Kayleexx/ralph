//! SQLite-backed session metadata store.
//!
//! Every state-changing write goes through one `rusqlite` transaction, relying on
//! SQLite's own WAL guarantees for crash-safety instead of a hand-rolled
//! temp-file+fsync+rename scheme.
use std::path::Path;

use rusqlite::{Connection, OptionalExtension, params};
use thiserror::Error;

use crate::state::SessionState;
use crate::typo::suggest_similar;

mod migrate;
#[cfg(test)]
mod tests;

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
        if let rusqlite::Error::SqliteFailure(inner, _) = &e
            && inner.code == rusqlite::ErrorCode::DatabaseBusy
        {
            return StorageError::Busy;
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
    updated_at          TEXT NOT NULL
)";

impl Storage {
    pub fn open(path: &Path) -> Result<Self, StorageError> {
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
                state, pid, location, token_count, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
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
            "UPDATE sessions SET model_revision = ?1, engine_version = ?2, updated_at = ?3 WHERE id = ?4",
            params![model_revision, engine_version, updated_at, id],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn set_token_count(
        &self,
        id: &str,
        token_count: i64,
        updated_at: &str,
    ) -> Result<(), StorageError> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "UPDATE sessions SET token_count = ?1, updated_at = ?2 WHERE id = ?3",
            params![token_count, updated_at, id],
        )?;
        tx.commit()?;
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
    pub fn resolve(&self, identifier: &str) -> Result<SessionRow, StorageError> {
        if let Some(row) = self.find_by_name(identifier)? {
            return Ok(row);
        }
        if let Some(row) = self.find_by_id(identifier)? {
            return Ok(row);
        }
        let matches = self.find_by_id_prefix(identifier)?;
        match matches.len() {
            0 => {
                let names = self.list()?.into_iter().map(|r| r.name).collect::<Vec<_>>();
                let suggestion = suggest_similar(identifier, names.iter().map(String::as_str));
                Err(StorageError::NotFound {
                    identifier: identifier.to_string(),
                    suggestion,
                })
            }
            1 => Ok(matches.into_iter().next().unwrap()),
            _ => Err(StorageError::AmbiguousPrefix(
                identifier.to_string(),
                matches.into_iter().map(|r| r.name).collect(),
            )),
        }
    }

    fn find_by_name(&self, name: &str) -> Result<Option<SessionRow>, StorageError> {
        self.conn
            .query_row(
                "SELECT * FROM sessions WHERE name = ?1",
                params![name],
                row_to_session,
            )
            .optional()
            .map_err(StorageError::from)
    }

    fn find_by_id(&self, id: &str) -> Result<Option<SessionRow>, StorageError> {
        self.conn
            .query_row(
                "SELECT * FROM sessions WHERE id = ?1",
                params![id],
                row_to_session,
            )
            .optional()
            .map_err(StorageError::from)
    }

    fn find_by_id_prefix(&self, prefix: &str) -> Result<Vec<SessionRow>, StorageError> {
        let pattern = format!("{prefix}%");
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM sessions WHERE id LIKE ?1")?;
        let rows = stmt.query_map(params![pattern], row_to_session)?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::from)
    }

    /// Any session left in `Starting`, or `Active` with no currently-live worker, is
    /// demoted to `Stopped` rather than left pretending to be usable. Called on daemon
    /// startup to reconcile state after a crash or reboot.
    pub fn reconcile_after_restart(
        &self,
        is_pid_alive: impl Fn(i64) -> bool,
        now: &str,
    ) -> Result<Vec<String>, StorageError> {
        let mut demoted = Vec::new();
        for row in self.list()? {
            let stale = match row.state {
                SessionState::Starting => true,
                SessionState::Active => match row.pid {
                    Some(pid) => !is_pid_alive(pid),
                    None => true,
                },
                _ => false,
            };
            if stale {
                self.set_state(&row.id, SessionState::Stopped, None, now)?;
                demoted.push(row.id);
            }
        }
        Ok(demoted)
    }
}

fn row_to_session(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionRow> {
    let state_str: String = row.get("state")?;
    let state = state_str.parse::<SessionState>().map_err(|e| {
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
    })
}
