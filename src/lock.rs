//! Daemon single-instance lock and per-session mutation locks.
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::{Arc, Mutex};

use fs2::FileExt;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

// The File is never read again, but must stay open for the process lifetime — dropping
// it releases the flock.
pub struct DaemonLock(#[allow(dead_code)] File);

impl DaemonLock {
    /// Non-blocking: fails immediately if another daemon already holds the lock, rather
    /// than queuing behind it.
    pub fn acquire(path: &Path) -> std::io::Result<Option<DaemonLock>> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        match file.try_lock_exclusive() {
            Ok(()) => Ok(Some(DaemonLock(file))),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }
}

/// Lazily-created per-session async mutexes. Holding a `tokio::sync::Mutex` guard across
/// an `.await` is the correct, idiomatic use of an async mutex — the "don't hold locks
/// across await" rule is about blocking `std::sync::Mutex`, not this.
#[derive(Default)]
pub struct SessionLocks {
    inner: Mutex<HashMap<String, Arc<AsyncMutex<()>>>>,
}

impl SessionLocks {
    pub fn new() -> Self {
        Self::default()
    }

    fn entry(&self, id: &str) -> Arc<AsyncMutex<()>> {
        let mut map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(id.to_string())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    }

    pub async fn acquire(&self, id: &str) -> OwnedMutexGuard<()> {
        self.entry(id).lock_owned().await
    }

    /// Fails fast rather than queuing, so a second concurrent mutating command gets a
    /// clear "operation already in progress" response instead of hanging.
    pub fn try_acquire(&self, id: &str) -> Option<OwnedMutexGuard<()>> {
        self.entry(id).try_lock_owned().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_daemon_lock_attempt_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon.lock");
        let first = DaemonLock::acquire(&path).unwrap();
        assert!(first.is_some());
        let second = DaemonLock::acquire(&path).unwrap();
        assert!(second.is_none());
    }

    #[test]
    fn lock_is_released_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon.lock");
        {
            let _first = DaemonLock::acquire(&path).unwrap();
        }
        let second = DaemonLock::acquire(&path).unwrap();
        assert!(second.is_some());
    }

    #[test]
    fn session_lock_contention_is_fast_fail() {
        let locks = SessionLocks::new();
        let guard = locks.try_acquire("demo");
        assert!(guard.is_some());
        let second = locks.try_acquire("demo");
        assert!(second.is_none());
        drop(guard);
        assert!(locks.try_acquire("demo").is_some());
    }

    #[test]
    fn different_sessions_do_not_contend() {
        let locks = SessionLocks::new();
        let a = locks.try_acquire("a");
        let b = locks.try_acquire("b");
        assert!(a.is_some());
        assert!(b.is_some());
    }
}
