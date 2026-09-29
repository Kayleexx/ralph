//! The compatibility fingerprint gating fast (native KV) vs. portable (full replay)
//! resume (RALPH_SPEC.md §5.4). Comparison is exact-match on every field: any difference
//! — a moved model revision, a different vLLM build, a different GPU — falls back to
//! portable reconstruction rather than risking a silent, wrong restore (Invariant 3).
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    pub model_revision: Option<String>,
    pub tokenizer_revision: Option<String>,
    pub engine: String,
    pub engine_version: Option<String>,
    /// From `nvidia-smi`, the same probe `daemon::admission` already uses — never a second
    /// GPU-detection path.
    pub gpu_name: Option<String>,
    /// Fixed at 1: nothing in this codebase runs tensor parallelism yet.
    pub tensor_parallel: u32,
    /// `None`: no LoRA/adapter support yet.
    pub adapter: Option<String>,
}

impl Fingerprint {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    pub fn from_json(json: &str) -> Option<Self> {
        serde_json::from_str(json).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Fingerprint {
        Fingerprint {
            model_revision: Some("abc".into()),
            tokenizer_revision: Some("abc".into()),
            engine: "vllm".into(),
            engine_version: Some("0.30.0".into()),
            gpu_name: Some("NVIDIA GeForce RTX 5050 Laptop GPU".into()),
            tensor_parallel: 1,
            adapter: None,
        }
    }

    #[test]
    fn round_trips_through_json() {
        let fp = sample();
        assert_eq!(Fingerprint::from_json(&fp.to_json()), Some(fp));
    }

    #[test]
    fn a_moved_revision_is_a_different_fingerprint() {
        let mut moved = sample();
        moved.model_revision = Some("different".into());
        assert_ne!(sample(), moved);
    }

    #[test]
    fn corrupt_json_never_panics() {
        assert_eq!(Fingerprint::from_json("not json"), None);
    }
}
