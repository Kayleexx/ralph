//! The compatibility fingerprint gating fast (native KV) vs. portable (full replay)
//! resume. Comparison is exact-match on every field: any difference — a moved model
//! revision, a different vLLM build, a different GPU — falls back to portable
//! reconstruction rather than risking a silent, wrong restore.
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

/// The reason behind `InspectInfo::fast_restore`, not just a yes/no.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RestoreReadiness {
    NativeAvailable,
    NoNativeState,
    NativeCorrupt,
    EngineMismatch,
    ModelMismatch,
    PortableAvailable,
}

impl RestoreReadiness {
    pub fn as_str(self) -> &'static str {
        match self {
            RestoreReadiness::NativeAvailable => "native available",
            RestoreReadiness::NoNativeState => "portable only - never checkpointed",
            RestoreReadiness::NativeCorrupt => "portable only - checkpoint content corrupt",
            RestoreReadiness::EngineMismatch => "portable only - engine mismatch",
            RestoreReadiness::ModelMismatch => "portable only - model/tokenizer mismatch",
            RestoreReadiness::PortableAvailable => {
                "portable only - checkpoint not usable right now"
            }
        }
    }
}

/// Identity mismatches are checked before content corruption: a mismatched fingerprint
/// means the content check doesn't apply to this session's KV at all.
pub fn classify(
    current: &Fingerprint,
    stored: Option<&Fingerprint>,
    content_intact: bool,
    room_ok: bool,
) -> RestoreReadiness {
    let Some(stored) = stored else {
        return RestoreReadiness::NoNativeState;
    };
    if stored.engine != current.engine || stored.engine_version != current.engine_version {
        return RestoreReadiness::EngineMismatch;
    }
    if stored.model_revision != current.model_revision
        || stored.tokenizer_revision != current.tokenizer_revision
        || stored.gpu_name != current.gpu_name
    {
        return RestoreReadiness::ModelMismatch;
    }
    if !content_intact {
        return RestoreReadiness::NativeCorrupt;
    }
    if !room_ok {
        return RestoreReadiness::PortableAvailable;
    }
    RestoreReadiness::NativeAvailable
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

    #[test]
    fn classify_reports_every_reason_distinctly() {
        let current = sample();
        assert_eq!(
            classify(&current, None, true, true),
            RestoreReadiness::NoNativeState
        );

        let mut engine_mismatch = sample();
        engine_mismatch.engine = "sglang".into();
        assert_eq!(
            classify(&current, Some(&engine_mismatch), true, true),
            RestoreReadiness::EngineMismatch
        );

        let mut model_mismatch = sample();
        model_mismatch.model_revision = Some("different".into());
        assert_eq!(
            classify(&current, Some(&model_mismatch), true, true),
            RestoreReadiness::ModelMismatch
        );

        assert_eq!(
            classify(&current, Some(&current), false, true),
            RestoreReadiness::NativeCorrupt
        );
        assert_eq!(
            classify(&current, Some(&current), true, false),
            RestoreReadiness::PortableAvailable
        );
        assert_eq!(
            classify(&current, Some(&current), true, true),
            RestoreReadiness::NativeAvailable
        );
    }
}
