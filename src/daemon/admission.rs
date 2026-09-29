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
    pub name: String,
    pub total_mib: u64,
    pub free_mib: u64,
    pub allocations: HashMap<u32, u64>,
}

pub(super) fn gpu_memory() -> Result<GpuMemory, CliError> {
    fn query(args: &[&str]) -> Result<String, CliError> {
        let out = std::process::Command::new("nvidia-smi")
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
    let gpu = query(&[
        "--id=0",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ])?;
    let fields: Vec<_> = gpu.trim().split(',').map(str::trim).collect();
    let bad = || {
        CliError::Resource(
            "unreadable GPU memory measurement; session preserved, retry ralph doctor".into(),
        )
    };
    if fields.len() != 3 {
        return Err(bad());
    }
    let apps = query(&[
        "--id=0",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])?;
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
    Ok(GpuMemory {
        name: fields[0].into(),
        total_mib: fields[1].parse().map_err(|_| bad())?,
        free_mib: fields[2].parse().map_err(|_| bad())?,
        allocations,
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
            .values()
            .filter_map(|w| w.profile.map(|p| (w.engine.clone(), p)))
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

    async fn available_vram(&self) -> Result<(u64, u64), CliError> {
        let gpu = (self.gpu_memory)()?;
        if gpu.name != "NVIDIA GeForce RTX 5050 Laptop GPU" || gpu.total_mib != 8151 {
            return Err(CliError::Resource(format!(
                "no measured safe profiles for {}; session preserved, use a measured GPU",
                gpu.name
            )));
        }
        let reserved = self.reserved_deficit(&gpu).await + HEADROOM_MIB;
        Ok((gpu.free_mib, reserved))
    }

    pub(super) async fn admitted_spec(&self, row: &SessionRow) -> Result<ModelSpec, CliError> {
        let saved = self.storage().worker_profile(&row.id)?;
        let other_unmeasured = self
            .workers
            .lock()
            .await
            .values()
            .any(|w| w.profile.is_none());
        let Some((revision, candidates)) = profiles::measured(&row.model) else {
            if saved.is_some() || !self.workers.lock().await.is_empty() {
                return Err(CliError::Resource(format!(
                    "{} has no measured safe co-residency profile; session preserved, use a measured Qwen model",
                    row.model
                )));
            }
            return Ok(ModelSpec {
                model: row.model.clone(),
                revision: row.model_revision.clone(),
                profile: None,
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
        let (free, reserved) = self.available_vram().await?;
        let profile = saved
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
            });
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
    ) -> Result<(), CliError> {
        if !engine.lock().await.is_sleeping().await {
            return Ok(());
        }
        if profile.is_some() {
            let (free, reserved) = self.available_vram().await?;
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
    ) -> Result<(), CliError> {
        if !engine.lock().await.is_sleeping().await {
            return Ok(());
        }
        let _startup = self.startup.lock().await;
        self.wake_reserved(engine, profile).await
    }
}
