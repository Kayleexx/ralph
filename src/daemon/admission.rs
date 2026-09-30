//! Static measured budgets; the existing startup guard serializes admission and wake.
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;

use super::Daemon;
use crate::engine::profiles::{self, HEADROOM_MIB, WorkerProfile};
use crate::engine::{Engine, ModelSpec};
use crate::error::CliError;
use crate::storage::SessionRow;

pub(super) struct GpuMemory {
    pub index: u32,
    pub name: String,
    pub free_mib: u64,
    pub allocations: HashMap<u32, u64>,
}

pub(super) fn gpu_memory(index: u32) -> Result<GpuMemory, CliError> {
    fn query(index: u32, args: &[&str]) -> Result<String, CliError> {
        let out = std::process::Command::new("nvidia-smi")
            .arg(format!("--id={index}"))
            .args(args)
            .output()
            .map_err(|e| {
                CliError::Resource(format!(
                    "cannot measure free VRAM: {e}; retry after checking ralph doctor"
                ))
            })?;
        if !out.status.success() {
            return Err(CliError::Resource(
                "cannot measure free VRAM; retry after checking ralph doctor".into(),
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }
    let gpu = query(
        index,
        &[
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let fields: Vec<_> = gpu.trim().split(',').map(str::trim).collect();
    let bad = || {
        CliError::Resource(
            "unreadable GPU memory measurement; session preserved, retry ralph doctor".into(),
        )
    };
    if fields.len() != 3 {
        return Err(bad());
    }
    let apps = query(
        index,
        &[
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let mut allocations = HashMap::new();
    for line in apps.lines().filter(|s| !s.trim().is_empty()) {
        let Some((pid, memory)) = line.split_once(',') else {
            return Err(bad());
        };
        allocations.insert(
            pid.trim().parse().map_err(|_| bad())?,
            memory.trim().parse().map_err(|_| bad())?,
        );
    }
    // `fields[1]` (memory.total) isn't used, but the query still asks for it so the
    // 3-field count check catches a genuinely unreadable line rather than one that
    // happens to parse with only 2.
    Ok(GpuMemory {
        index,
        name: fields[0].into(),
        free_mib: fields[2].parse().map_err(|_| bad())?,
        allocations,
    })
}

fn pick_profile(
    saved: Option<WorkerProfile>,
    candidates: &[WorkerProfile],
    free: u64,
    reserved: u64,
) -> Option<WorkerProfile> {
    saved
        .filter(|p| p.reservation_mib() + reserved <= free)
        .or_else(|| {
            if saved.is_none() {
                candidates
                    .iter()
                    .copied()
                    .find(|p| p.reservation_mib() + reserved <= free)
            } else {
                None
            }
        })
}

fn group(pid: u32) -> Option<u32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    stat.rsplit_once(')')?
        .1
        .split_whitespace()
        .nth(2)?
        .parse()
        .ok()
}

impl<E: Engine + 'static> Daemon<E> {
    async fn reserved_deficit(&self, gpu: &GpuMemory) -> u64 {
        let entries: Vec<_> = self
            .workers
            .lock()
            .await
            .iter()
            .filter(|(key, _)| key.gpu == gpu.index)
            .filter_map(|(_, w)| w.profile.map(|p| (w.engine.clone(), p)))
            .collect();
        let mut deficit = 0;
        for (engine, profile) in entries {
            let pid = engine.lock().await.pid();
            let allocated: u64 = gpu
                .allocations
                .iter()
                .filter(|(p, _)| {
                    pid.is_some_and(|leader| **p == leader || group(**p) == Some(leader))
                })
                .map(|(_, bytes)| bytes)
                .sum();
            // Sleep releases weights/KV but keeps the full logical wake reservation.
            deficit += profile.reservation_mib().saturating_sub(allocated);
        }
        deficit
    }

    // `gpu_memory(index)` (an injectable fn pointer — real callers hit `nvidia-smi
    // --id=<index>`, which itself fails cleanly with a nonzero exit for an out-of-range
    // index) is the one and only visibility check — no separate `gpu_count()`
    // pre-check, so tests can simulate an unavailable GPU by overriding `gpu_memory`
    // rather than needing real multi-GPU hardware.
    async fn available_vram(&self, gpu_index: u32) -> Result<(u64, u64), CliError> {
        let gpu = (self.gpu_memory)(gpu_index)?;
        let reserved = self.reserved_deficit(&gpu).await + HEADROOM_MIB;
        Ok((gpu.free_mib, reserved))
    }

    pub(super) async fn admitted_spec(
        self: &Arc<Self>,
        row: &SessionRow,
        allow_native: bool,
        gpu_index: u32,
    ) -> Result<ModelSpec, CliError> {
        let kv_offload = allow_native.then(|| self.kv_offload_for(row)).flatten();
        let saved = self.storage().worker_profile(&row.id)?;
        let other_unmeasured = self
            .workers
            .lock()
            .await
            .iter()
            .filter(|(key, _)| key.gpu == gpu_index)
            .any(|(_, w)| w.profile.is_none());
        let Some((revision, candidates)) = profiles::measured(&row.model) else {
            let any_on_this_gpu = self
                .workers
                .lock()
                .await
                .keys()
                .any(|key| key.gpu == gpu_index);
            if saved.is_some() || any_on_this_gpu {
                return Err(CliError::Resource(format!(
                    "{} has no measured safe co-residency profile; session preserved, use a measured Qwen model",
                    row.model
                )));
            }
            return Ok(ModelSpec {
                model: row.model.clone(),
                revision: row.model_revision.clone(),
                profile: None,
                kv_offload,
                gpu: gpu_index,
            });
        };
        if other_unmeasured {
            return Err(CliError::Resource("resident worker has no measured wake budget; session preserved, run the Qwen models in an isolated daemon".into()));
        }
        if row.model_revision.as_deref().is_some_and(|r| r != revision)
            || row
                .tokenizer_revision
                .as_deref()
                .is_some_and(|r| r != revision)
            || saved.is_some_and(|p| !candidates.contains(&p))
        {
            return Err(CliError::InvalidState(
                "stored model identity or budget differs from measured profiles; history preserved"
                    .into(),
            ));
        }
        // The GPU driver can lag slightly reclaiming VRAM after a worker process exits
        // (e.g. right after `ralph pause` stops one and a resume immediately follows) —
        // a bounded retry here is measuring a real, transient external resource, not
        // papering over a logic bug (unlike a sleep-based test wait).
        let mut attempt = 0;
        let (free, reserved, profile) = loop {
            let (free, reserved) = self.available_vram(gpu_index).await?;
            let profile = pick_profile(saved, candidates, free, reserved);
            attempt += 1;
            if profile.is_some() || attempt >= 5 {
                break (free, reserved, profile);
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        };
        // Every retry above only absorbs driver VRAM-reclaim lag; nothing changes the
        // *actual* free VRAM. If it's still not enough, try demoting one other resident
        // session once (`make_room`) before giving up — the one real "make room" attempt
        // this phase performs, never touching the session being admitted itself. The
        // freed VRAM is subject to the exact same driver-reclaim lag as above (confirmed
        // live: a single immediate re-check right after `make_room`'s own `kill -KILL`
        // can still read the pre-eviction free figure), so this re-check gets the same
        // bounded retry, not a one-shot look.
        let (free, reserved, profile) = if profile.is_none() {
            match self.make_room(gpu_index, &row.id).await {
                Some(_freed) => {
                    let mut attempt = 0;
                    loop {
                        let (free, reserved) = self.available_vram(gpu_index).await?;
                        let profile = pick_profile(saved, candidates, free, reserved);
                        attempt += 1;
                        if profile.is_some() || attempt >= 5 {
                            break (free, reserved, profile);
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    }
                }
                None => (free, reserved, profile),
            }
        } else {
            (free, reserved, profile)
        };
        let Some(profile) = profile else {
            let minimum = saved
                .or_else(|| candidates.last().copied())
                .ok_or_else(|| CliError::Resource("no measured profiles available".into()))?;
            return Err(CliError::Resource(format!(
                "cannot start {} safely: {} MiB free, {} MiB required including worker/wake reserves and 1024 MiB headroom; session preserved, free VRAM and retry",
                row.model,
                free,
                minimum.reservation_mib() + reserved
            )));
        };
        self.storage().set_worker_profile(&row.id, profile)?;
        Ok(ModelSpec {
            model: row.model.clone(),
            revision: Some(revision.into()),
            profile: Some(profile),
            kv_offload,
            gpu: gpu_index,
        })
    }

    pub(super) fn share_profile(
        &self,
        id: &str,
        profile: Option<WorkerProfile>,
    ) -> Result<(), CliError> {
        let saved = self.storage().worker_profile(id)?;
        if saved.is_some() && saved != profile {
            return Err(CliError::InvalidState(
                "resident worker has a different context/KV budget; history preserved".into(),
            ));
        }
        if let Some(profile) = profile {
            self.storage().set_worker_profile(id, profile)?;
        }
        Ok(())
    }

    // Caller owns startup; engine and registry locks are never nested.
    pub(super) async fn wake_reserved(
        &self,
        engine: &Arc<AsyncMutex<E>>,
        profile: Option<WorkerProfile>,
        gpu_index: u32,
    ) -> Result<(), CliError> {
        if !engine.lock().await.is_sleeping().await {
            return Ok(());
        }
        if profile.is_some() {
            let (free, reserved) = self.available_vram(gpu_index).await?;
            if free < reserved {
                return Err(CliError::Resource(format!(
                    "cannot wake worker safely: {free} MiB free, {reserved} MiB required for wake reserves and headroom; session preserved, free VRAM and retry"
                )));
            }
        }
        engine
            .lock()
            .await
            .wake_up()
            .await
            .map_err(super::map_engine_error)
    }

    pub(super) async fn wake_worker(
        &self,
        engine: &Arc<AsyncMutex<E>>,
        profile: Option<WorkerProfile>,
        gpu_index: u32,
    ) -> Result<(), CliError> {
        if !engine.lock().await.is_sleeping().await {
            return Ok(());
        }
        let _startup = self.startup.lock().await;
        self.wake_reserved(engine, profile, gpu_index).await
    }
}
