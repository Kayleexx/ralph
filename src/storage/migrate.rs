//! Versioned schema migrations, run on every `Storage::open`. Editing `SCHEMA`'s
//! `CREATE TABLE IF NOT EXISTS` text alone does nothing for a database file that already
//! exists with the old shape — that exact gap once turned a newly-nullable column into
//! an on-disk `NOT NULL` violation, which got misreported as a duplicate session name.
//! Every future schema change adds an entry to `MIGRATIONS` instead of relying on
//! `CREATE TABLE IF NOT EXISTS` to somehow fix up existing rows.
use rusqlite::{Connection, OptionalExtension};

use super::{SCHEMA, StorageError};

type Migration = fn(&Connection) -> Result<(), StorageError>;

const MIGRATIONS: &[Migration] = &[
    migrate_v1_nullable_model_revision,
    migrate_v2_token_turns,
    migrate_v3_restarts,
    migrate_v4_tokens,
    migrate_v5_integrity,
    migrate_v6_profiles,
    migrate_v7_checkpoints,
];

pub(crate) fn run(conn: &Connection) -> Result<(), StorageError> {
    let current: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    for (i, migration) in MIGRATIONS.iter().enumerate().skip(current.max(0) as usize) {
        let tx = conn.unchecked_transaction()?;
        migration(&tx)?;
        tx.pragma_update(None, "user_version", (i + 1) as i64)?;
        tx.commit()?;
    }
    Ok(())
}

fn table_exists(conn: &Connection, name: &str) -> Result<bool, StorageError> {
    let count: i64 = conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
        [name],
        |r| r.get(0),
    )?;
    Ok(count > 0)
}

/// `model_revision` used to be `NOT NULL`; a session's revision is now legitimately
/// unresolved (`NULL`) until `start_model` succeeds, so a database created by an older
/// build needs its `sessions` table rebuilt with that column made nullable. A brand-new
/// database has no `sessions` table yet — nothing to migrate, since `Storage::open`
/// creates it fresh with the current (already-nullable) schema right after this runs.
fn migrate_v1_nullable_model_revision(conn: &Connection) -> Result<(), StorageError> {
    if !table_exists(conn, "sessions")? {
        return Ok(());
    }
    let notnull: Option<i64> = conn
        .query_row(
            "SELECT \"notnull\" FROM pragma_table_info('sessions') WHERE name = 'model_revision'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    if notnull != Some(1) {
        return Ok(()); // already nullable
    }
    conn.execute_batch(&format!(
        "ALTER TABLE sessions RENAME TO sessions_v0;
         {SCHEMA};
         INSERT INTO sessions (
             id, name, model, model_revision, tokenizer_revision, engine, engine_version,
             state, pid, location, token_count, created_at, updated_at
         )
         SELECT
             id, name, model, model_revision, tokenizer_revision, engine, engine_version,
             state, pid, location, token_count, created_at, updated_at
         FROM sessions_v0;
         DROP TABLE sessions_v0;"
    ))?;
    Ok(())
}

/// `token_turns` (durable turn history) didn't exist before this migration —
/// `CREATE TABLE IF NOT EXISTS` alone is enough here since there's no existing data to
/// reshape, unlike `migrate_v1_nullable_model_revision`.
fn migrate_v2_token_turns(conn: &Connection) -> Result<(), StorageError> {
    conn.execute_batch(super::token_log::SCHEMA)?;
    Ok(())
}

fn migrate_v3_restarts(conn: &Connection) -> Result<(), StorageError> {
    conn.execute_batch("CREATE TABLE worker_restarts (model TEXT PRIMARY KEY, attempts INTEGER NOT NULL, failure TEXT NOT NULL)")?;
    Ok(())
}

fn migrate_v4_tokens(conn: &Connection) -> Result<(), StorageError> {
    conn.execute_batch("ALTER TABLE token_turns ADD COLUMN token_ids TEXT; ALTER TABLE token_turns ADD COLUMN complete INTEGER")?;
    Ok(())
}

fn migrate_v5_integrity(conn: &Connection) -> Result<(), StorageError> {
    conn.execute_batch("ALTER TABLE token_turns ADD COLUMN checksum TEXT")?;
    super::token_log::seal_existing(conn)?;
    Ok(())
}

fn migrate_v6_profiles(conn: &Connection) -> Result<(), StorageError> {
    conn.execute_batch(
        "CREATE TABLE session_profiles (session_id TEXT PRIMARY KEY, profile TEXT NOT NULL)",
    )?;
    Ok(())
}

fn migrate_v7_checkpoints(conn: &Connection) -> Result<(), StorageError> {
    conn.execute_batch(super::checkpoints::SCHEMA)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn old_schema_db_with_one_row() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                model TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                tokenizer_revision TEXT,
                engine TEXT NOT NULL,
                engine_version TEXT,
                state TEXT NOT NULL,
                pid INTEGER,
                location TEXT NOT NULL DEFAULT 'local/gpu0',
                token_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO sessions (id, name, model, model_revision, engine, state, location, token_count, created_at, updated_at)
            VALUES ('01AAA', 'demo', 'some/model', 'some/model', 'vllm', 'stopped', 'local/gpu0', 0, 't', 't');",
        )
        .unwrap();
        conn
    }

    #[test]
    fn migrates_old_not_null_schema_and_keeps_existing_data() {
        let conn = old_schema_db_with_one_row();
        run(&conn).unwrap();

        let notnull: i64 = conn
            .query_row("SELECT \"notnull\" FROM pragma_table_info('sessions') WHERE name = 'model_revision'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(notnull, 0, "model_revision should now be nullable");

        let name: String = conn
            .query_row("SELECT name FROM sessions WHERE id = '01AAA'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(name, "demo", "existing row must survive the migration");
    }

    #[test]
    fn inserting_a_null_revision_succeeds_after_migration() {
        let conn = old_schema_db_with_one_row();
        run(&conn).unwrap();

        conn.execute(
            "INSERT INTO sessions (id, name, model, model_revision, engine, state, location, token_count, created_at, updated_at)
             VALUES ('01BBB', 'brand-new', 'some/model', NULL, 'vllm', 'created', 'local/gpu0', 0, 't', 't')",
            [],
        )
        .unwrap();
    }

    #[test]
    fn fresh_database_has_no_sessions_table_to_migrate() {
        let conn = Connection::open_in_memory().unwrap();
        run(&conn).unwrap(); // must not error just because the table doesn't exist yet
    }

    #[test]
    fn running_migrations_twice_is_a_no_op() {
        let conn = old_schema_db_with_one_row();
        run(&conn).unwrap();
        run(&conn).unwrap(); // must not try to migrate an already-migrated table again
        let name: String = conn
            .query_row("SELECT name FROM sessions WHERE id = '01AAA'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(name, "demo");
    }
}
