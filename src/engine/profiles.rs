//! Measured BF16/eager profiles for the RTX 5050 Laptop (vLLM 0.30.0).
use serde::{Deserialize, Serialize};

pub const HEADROOM_MIB: u64 = 1024;
pub const ALLOWANCE_MIB: u64 = 150;
pub const QWEN25: &str = "Qwen/Qwen2.5-0.5B-Instruct";
pub const QWEN3: &str = "Qwen/Qwen3-0.6B";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerProfile {
    pub max_context: u32,
    pub kv_mib: u32,
    pub peak_mib: u32,
}

impl WorkerProfile {
    pub fn reservation_mib(self) -> u64 {
        u64::from(self.peak_mib) + ALLOWANCE_MIB
    }
}

pub fn measured(model: &str) -> Option<(&'static str, &'static [WorkerProfile])> {
    match model {
        QWEN25 => Some((
            "7ae557604adf67be50417f59c2c2f167def9a775",
            &[
                WorkerProfile {
                    max_context: 16384,
                    kv_mib: 448,
                    peak_mib: 2142,
                },
                WorkerProfile {
                    max_context: 8192,
                    kv_mib: 256,
                    peak_mib: 1950,
                },
            ],
        )),
        QWEN3 => Some((
            "c1899de289a04d12100db370d81485cdf75e47ca",
            &[
                WorkerProfile {
                    max_context: 4096,
                    kv_mib: 960,
                    peak_mib: 2864,
                },
                WorkerProfile {
                    max_context: 2048,
                    kv_mib: 512,
                    peak_mib: 2416,
                },
            ],
        )),
        _ => None,
    }
}
