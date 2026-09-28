//! Cumulative turn writes and session counts share one WAL transaction.
use super::{Storage, StorageError};
use crate::engine::{ChatMessage, Role};
use rusqlite::{Connection, OptionalExtension, params};

pub(super) const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS token_turns (
    session_id TEXT NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, token_count INTEGER NOT NULL, committed_at TEXT NOT NULL,
    PRIMARY KEY (session_id, seq))";

// Stable FNV-1a catches accidental payload corruption after WAL checkpointing.
fn checksum(
    seq: i64,
    role: &str,
    content: &str,
    count: i64,
    ids: &str,
    complete: i64,
) -> Result<String, StorageError> {
    let bytes = serde_json::to_vec(&(seq, role, content, count, ids, complete))
        .map_err(|e| StorageError::Corrupt(e.to_string()))?;
    let hash = bytes.into_iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    Ok(format!("{hash:016x}"))
}

pub(super) fn seal_existing(conn: &Connection) -> Result<(), StorageError> {
    let mut statement = conn.prepare(
        "SELECT session_id,seq,role,content,token_count,token_ids,complete FROM token_turns",
    )?;
    let rows = statement.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, i64>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, i64>(4)?,
            r.get::<_, Option<String>>(5)?,
            r.get::<_, Option<i64>>(6)?,
        ))
    })?;
    for row in rows {
        let (session, seq, role, content, count, ids, complete) = row?;
        // Legacy rows without token data remain incomplete; no history is invented.
        if let (Some(ids), Some(complete)) = (ids, complete) {
            conn.execute(
                "UPDATE token_turns SET checksum=?3 WHERE session_id=?1 AND seq=?2",
                params![
                    session,
                    seq,
                    checksum(seq, &role, &content, count, &ids, complete)?
                ],
            )?;
        }
    }
    Ok(())
}

fn update_counts(conn: &Connection, session: &str, now: &str) -> Result<(), StorageError> {
    conn.execute("UPDATE sessions SET token_count=(SELECT COALESCE(SUM(token_count),0) FROM token_turns WHERE session_id=?1), updated_at=?2 WHERE id=?1", params![session, now])?;
    Ok(())
}

impl Storage {
    pub fn record_user_turn(
        &self,
        session: &str,
        content: &str,
        ids: &[u32],
        now: &str,
    ) -> Result<i64, StorageError> {
        self.replay_turns(session)?;
        let tx = self.conn.unchecked_transaction()?;
        let seq: i64 = tx.query_row(
            "SELECT COALESCE(MAX(seq)+1,0) FROM token_turns WHERE session_id=?1",
            [session],
            |r| r.get(0),
        )?;
        let encoded =
            serde_json::to_string(ids).map_err(|e| StorageError::Corrupt(e.to_string()))?;
        tx.execute(
            "INSERT INTO token_turns VALUES (?1,?2,'user',?3,?4,?5,?6,1,?7)",
            params![
                session,
                seq,
                content,
                ids.len() as i64,
                now,
                encoded,
                checksum(seq, "user", content, ids.len() as i64, &encoded, 1)?
            ],
        )?;
        tx.execute(
            "INSERT INTO token_turns VALUES (?1,?2,'assistant','',0,?3,'[]',0,?4)",
            params![
                session,
                seq + 1,
                now,
                checksum(seq + 1, "assistant", "", 0, "[]", 0)?
            ],
        )?;
        update_counts(&tx, session, now)?;
        tx.commit()?;
        Ok(seq + 1)
    }

    // ponytail: cumulative replies rewrite prior output; use chunks if long replies make flushes expensive.
    pub fn flush_assistant_turn(
        &self,
        session: &str,
        seq: i64,
        content: &str,
        ids: &[u32],
        now: &str,
    ) -> Result<(), StorageError> {
        let tx = self.conn.unchecked_transaction()?;
        let old = tx.query_row("SELECT role,content,token_ids,complete,token_count,checksum FROM token_turns WHERE session_id=?1 AND seq=?2", params![session, seq], |r| Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,i64>(3)?,r.get::<_,i64>(4)?,r.get::<_,Option<String>>(5)?))).optional()?;
        let Some((role, committed, old_ids, complete, old_count, old_checksum)) = old else {
            return Err(StorageError::Corrupt(
                "assistant reservation missing".into(),
            ));
        };
        if old_checksum.as_deref()
            != Some(checksum(seq, &role, &committed, old_count, &old_ids, complete)?.as_str())
        {
            return Err(StorageError::Corrupt("turn checksum mismatch".into()));
        }
        let committed_ids: Vec<u32> = serde_json::from_str(&old_ids)
            .map_err(|_| StorageError::Corrupt("invalid token IDs".into()))?;
        if old_count != committed_ids.len() as i64
            || !matches!(complete, 0 | 1)
            || role != "assistant"
            || !content.starts_with(&committed)
            || !ids.starts_with(&committed_ids)
            || (complete == 1 && (content != committed || ids != committed_ids))
        {
            return Err(StorageError::Corrupt(
                "assistant flush differs from committed prefix".into(),
            ));
        }
        let encoded =
            serde_json::to_string(ids).map_err(|e| StorageError::Corrupt(e.to_string()))?;
        tx.execute("UPDATE token_turns SET content=?3,token_count=?4,committed_at=?5,token_ids=?6,checksum=?7 WHERE session_id=?1 AND seq=?2", params![session,seq,content,ids.len() as i64,now,encoded,checksum(seq,"assistant",content,ids.len() as i64,&encoded,complete)?])?;
        update_counts(&tx, session, now)?;
        tx.commit()?;
        Ok(())
    }

    pub fn finish_turn(&self, session: &str, seq: i64, complete: bool) -> Result<(), StorageError> {
        let tx = self.conn.unchecked_transaction()?;
        let (role,content,count,ids,old_complete,old_checksum)=tx.query_row("SELECT role,content,token_count,token_ids,complete,checksum FROM token_turns WHERE session_id=?1 AND seq=?2",params![session,seq],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,i64>(2)?,r.get::<_,String>(3)?,r.get::<_,i64>(4)?,r.get::<_,Option<String>>(5)?)))?;
        if role != "assistant"
            || old_checksum.as_deref()
                != Some(checksum(seq, &role, &content, count, &ids, old_complete)?.as_str())
        {
            return Err(StorageError::Corrupt(
                "assistant completion checksum mismatch".into(),
            ));
        }
        tx.execute(
            "UPDATE token_turns SET complete=?3,checksum=?4 WHERE session_id=?1 AND seq=?2",
            params![
                session,
                seq,
                i64::from(complete),
                checksum(seq, &role, &content, count, &ids, i64::from(complete))?
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn replay_turns(&self, session: &str) -> Result<Vec<ChatMessage>, StorageError> {
        let mut stmt = self.conn.prepare("SELECT seq,role,content,token_count,token_ids,complete,checksum FROM token_turns WHERE session_id=?1 ORDER BY seq")?;
        let rows = stmt.query_map([session], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<i64>>(5)?,
                r.get::<_, Option<String>>(6)?,
            ))
        })?;
        let mut messages = Vec::new();
        let mut total = 0;
        let mut turns = 0;
        for (expected, row) in rows.enumerate() {
            let (seq, role, content, count, ids, complete, stored_checksum) = row?;
            let expected_role = if seq % 2 == 0 { "user" } else { "assistant" };
            if seq != expected as i64
                || role != expected_role
                || count < 0
                || !matches!(complete, Some(0 | 1))
                || (role == "user" && complete != Some(1))
            {
                return Err(StorageError::Corrupt(
                    "invalid turn ordering, role or completion status".into(),
                ));
            }
            if stored_checksum.as_deref()
                != Some(
                    checksum(
                        seq,
                        &role,
                        &content,
                        count,
                        ids.as_deref().unwrap_or(""),
                        complete.unwrap_or(0),
                    )?
                    .as_str(),
                )
            {
                return Err(StorageError::Corrupt(
                    "turn checksum mismatch or missing legacy integrity data".into(),
                ));
            }
            let ids: Vec<u32> =
                serde_json::from_str(ids.as_deref().ok_or_else(|| {
                    StorageError::Corrupt("legacy history lacks token data".into())
                })?)
                .map_err(|_| StorageError::Corrupt("invalid token IDs".into()))?;
            if ids.len() as i64 != count {
                return Err(StorageError::Corrupt(
                    "token count differs from durable IDs".into(),
                ));
            }
            total += count;
            turns += 1;
            if role == "user" || !content.is_empty() {
                messages.push(ChatMessage {
                    role: if role == "user" {
                        Role::User
                    } else {
                        Role::Assistant
                    },
                    content,
                });
            }
        }
        if turns % 2 != 0 {
            return Err(StorageError::Corrupt(
                "incomplete accepted turn reservation".into(),
            ));
        }
        let stored: Option<i64> = self
            .conn
            .query_row(
                "SELECT token_count FROM sessions WHERE id=?1",
                [session],
                |r| r.get(0),
            )
            .optional()?;
        if stored.is_some_and(|count| count != total) {
            return Err(StorageError::Corrupt(
                "session count differs from durable history; legacy state may be incomplete".into(),
            ));
        }
        Ok(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cumulative_flush_is_idempotent_and_preserves_prefix() {
        let storage = Storage::open_in_memory().unwrap();
        let seq = storage.record_user_turn("s", "hello", &[1], "t").unwrap();
        storage
            .flush_assistant_turn("s", seq, "hi", &[2], "t")
            .unwrap();
        storage
            .flush_assistant_turn("s", seq, "hi", &[2], "t")
            .unwrap();
        storage
            .flush_assistant_turn("s", seq, "hi there", &[2, 3], "t")
            .unwrap();
        assert_eq!(storage.replay_turns("s").unwrap()[1].content, "hi there");
        assert!(
            storage
                .flush_assistant_turn("s", seq, "changed", &[2], "t")
                .is_err()
        );
    }
    #[test]
    fn corruption_is_rejected() {
        let storage = Storage::open_in_memory().unwrap();
        storage.record_user_turn("s", "hello", &[1], "t").unwrap();
        storage
            .conn
            .execute("UPDATE token_turns SET role='invalid' WHERE seq=0", [])
            .unwrap();
        assert!(matches!(
            storage.replay_turns("s"),
            Err(StorageError::Corrupt(_))
        ));
    }
    #[test]
    fn empty_input_is_retained_and_missing_reservation_is_corrupt() {
        let storage = Storage::open_in_memory().unwrap();
        storage.record_user_turn("s", "", &[], "t").unwrap();
        assert_eq!(storage.replay_turns("s").unwrap().len(), 1);
        storage
            .conn
            .execute("DELETE FROM token_turns WHERE seq=1", [])
            .unwrap();
        assert!(matches!(
            storage.replay_turns("s"),
            Err(StorageError::Corrupt(_))
        ));
    }
    #[test]
    fn valid_looking_payload_corruption_is_rejected_without_repair() {
        for sql in [
            "UPDATE token_turns SET content='hullo' WHERE seq=0",
            "UPDATE token_turns SET token_ids='[2]' WHERE seq=0",
            "UPDATE token_turns SET complete=1 WHERE seq=1",
        ] {
            let storage = Storage::open_in_memory().unwrap();
            storage.record_user_turn("s", "hello", &[1], "t").unwrap();
            storage.conn.execute(sql, []).unwrap();
            assert!(matches!(
                storage.replay_turns("s"),
                Err(StorageError::Corrupt(_))
            ));
        }
    }
    #[test]
    fn interrupted_transaction_keeps_committed_history() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let storage = Storage::open(&path).unwrap();
        let seq = storage.record_user_turn("s", "hello", &[1], "t").unwrap();
        storage
            .flush_assistant_turn("s", seq, "hi", &[2], "t")
            .unwrap();
        {
            let tx = storage.conn.unchecked_transaction().unwrap();
            tx.execute("UPDATE token_turns SET content='uncommitted'", [])
                .unwrap();
        }
        drop(storage);
        let storage = Storage::open(&path).unwrap();
        assert_eq!(storage.replay_turns("s").unwrap()[1].content, "hi");
    }
}

#[cfg(test)]
mod crash_tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn sigkill_during_wal_transaction_keeps_only_committed_output() {
        if let Ok(root) = std::env::var("RALPH_WAL_CRASH_TEST") {
            let storage =
                Storage::open(std::path::Path::new(&root).join("test.db").as_path()).unwrap();
            let tx = storage.conn.unchecked_transaction().unwrap();
            tx.execute("UPDATE token_turns SET content='uncommitted'", [])
                .unwrap();
            std::fs::write(std::path::Path::new(&root).join("ready"), b"ready").unwrap();
            loop {
                std::thread::park();
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let storage = Storage::open(&path).unwrap();
        let seq = storage.record_user_turn("s", "hello", &[1], "t").unwrap();
        storage
            .flush_assistant_turn("s", seq, "hi", &[2], "t")
            .unwrap();
        drop(storage);
        let mut child=std::process::Command::new(std::env::current_exe().unwrap()).args(["--exact","storage::token_log::crash_tests::sigkill_during_wal_transaction_keeps_only_committed_output"]).env("RALPH_WAL_CRASH_TEST",dir.path()).stdout(std::process::Stdio::null()).spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !dir.path().join("ready").exists() {
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("transaction did not begin");
            }
            std::thread::yield_now();
        }
        child.kill().unwrap();
        child.wait().unwrap();
        let storage = Storage::open(&path).unwrap();
        assert_eq!(storage.replay_turns("s").unwrap()[1].content, "hi");
    }
}
