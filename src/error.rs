//! Domain errors, exit-code mapping, and the shared human-readable error envelope every
//! command's failure path renders through — this is what keeps error output consistent
//! (what failed, what state things are in, what to do next) without re-typing it per
//! command.
use thiserror::Error;

use crate::state::StateError;
use crate::storage::StorageError;

#[derive(Debug, Error)]
pub enum CliError {
    #[error("{0}")]
    Usage(String),
    #[error("session {0:?} already exists")]
    DuplicateName(String),
    #[error("no session found matching {identifier:?}")]
    NotFound {
        identifier: String,
        suggestion: Option<String>,
    },
    #[error("{0:?} matches more than one session: {1:?}")]
    AmbiguousPrefix(String, Vec<String>),
    #[error("session is not in a state that allows this: {0}")]
    InvalidState(String),
    #[error("resource unavailable: {0}")]
    Resource(String),
    #[error("persistent state is corrupt: {0}")]
    Corrupt(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<StorageError> for CliError {
    fn from(e: StorageError) -> Self {
        match e {
            StorageError::DuplicateName(name) => CliError::DuplicateName(name),
            StorageError::NotFound {
                identifier,
                suggestion,
            } => CliError::NotFound {
                identifier,
                suggestion,
            },
            StorageError::AmbiguousPrefix(id, matches) => CliError::AmbiguousPrefix(id, matches),
            StorageError::Busy => CliError::Resource("storage is busy, try again".to_string()),
            StorageError::Corrupt(detail) => CliError::Corrupt(detail),
            StorageError::Sqlite(e) => CliError::Other(anyhow::anyhow!(e)),
        }
    }
}

impl From<StateError> for CliError {
    fn from(e: StateError) -> Self {
        CliError::InvalidState(e.to_string())
    }
}

pub fn exit_code(err: &CliError) -> i32 {
    match err {
        CliError::Usage(_) | CliError::DuplicateName(_) => 2,
        CliError::NotFound { .. } | CliError::AmbiguousPrefix(_, _) => 3,
        CliError::InvalidState(_) => 4,
        CliError::Resource(_) => 6,
        CliError::Corrupt(_) => 8,
        CliError::Other(_) => 1,
    }
}

/// what failed / what state things are in / what to do next.
pub struct Envelope {
    pub summary: String,
    pub detail: Vec<String>,
    pub next: Option<String>,
}

impl Envelope {
    pub fn render(&self, ascii_only: bool) -> String {
        let mark = if ascii_only { "x" } else { "×" };
        let arrow = if ascii_only { "->" } else { "→" };
        let mut out = format!("{mark} {}", self.summary);
        for line in &self.detail {
            out.push_str(&format!("\n  {line}"));
        }
        if let Some(next) = &self.next {
            out.push_str(&format!("\n{arrow} {next}"));
        }
        out
    }
}

pub fn envelope(err: &CliError) -> Envelope {
    match err {
        CliError::Usage(msg) => Envelope {
            summary: msg.clone(),
            detail: vec![],
            next: None,
        },
        CliError::DuplicateName(name) => Envelope {
            summary: format!("session {name:?} already exists"),
            detail: vec![],
            next: Some(format!("use a different --name, or: ralph inspect {name}")),
        },
        CliError::NotFound {
            identifier,
            suggestion,
        } => Envelope {
            summary: format!("no session found matching {identifier:?}"),
            detail: vec![],
            next: Some(match suggestion {
                Some(name) => format!("did you mean: {name}?"),
                None => "run: ralph ps".to_string(),
            }),
        },
        CliError::AmbiguousPrefix(id, matches) => Envelope {
            summary: format!("{id:?} matches more than one session"),
            detail: matches.clone(),
            next: Some("use the full name or id".to_string()),
        },
        CliError::InvalidState(reason) => Envelope {
            summary: "session cannot do this right now".to_string(),
            detail: vec![reason.clone()],
            next: None,
        },
        CliError::Resource(reason) => Envelope {
            summary: reason.clone(),
            detail: vec![],
            next: Some("run: ralph doctor".to_string()),
        },
        CliError::Corrupt(reason) => Envelope {
            summary: "persistent state is corrupt".to_string(),
            detail: vec![reason.clone()],
            next: None,
        },
        CliError::Other(e) => Envelope {
            summary: "operation failed".to_string(),
            detail: vec![e.to_string()],
            next: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_match_the_documented_table() {
        assert_eq!(exit_code(&CliError::DuplicateName("x".into())), 2);
        assert_eq!(
            exit_code(&CliError::NotFound {
                identifier: "x".into(),
                suggestion: None
            }),
            3
        );
        assert_eq!(exit_code(&CliError::AmbiguousPrefix("x".into(), vec![])), 3);
        assert_eq!(exit_code(&CliError::InvalidState("x".into())), 4);
        assert_eq!(exit_code(&CliError::Resource("x".into())), 6);
        assert_eq!(exit_code(&CliError::Corrupt("x".into())), 8);
    }

    #[test]
    fn envelope_render_uses_ascii_fallback() {
        let err = CliError::DuplicateName("demo".to_string());
        let env = envelope(&err);
        let unicode = env.render(false);
        let ascii = env.render(true);
        assert!(unicode.starts_with('×'));
        assert!(ascii.starts_with('x'));
        assert!(unicode.contains('→'));
        assert!(ascii.contains("->"));
    }

    #[test]
    fn not_found_with_suggestion_hints_did_you_mean() {
        let err = CliError::NotFound {
            identifier: "dem".to_string(),
            suggestion: Some("demo".to_string()),
        };
        let env = envelope(&err);
        assert_eq!(env.next.as_deref(), Some("did you mean: demo?"));
    }

    #[test]
    fn not_found_without_suggestion_hints_ps() {
        let err = CliError::NotFound {
            identifier: "nope".to_string(),
            suggestion: None,
        };
        let env = envelope(&err);
        assert_eq!(env.next.as_deref(), Some("run: ralph ps"));
    }

    #[test]
    fn storage_errors_map_to_cli_errors() {
        let err: CliError = StorageError::DuplicateName("demo".into()).into();
        assert!(matches!(err, CliError::DuplicateName(_)));
        assert_eq!(exit_code(&err), 2);
    }
}
