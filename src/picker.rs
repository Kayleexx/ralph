//! Interactive model + session-name picker for `ralph run` when invoked with no model on
//! a real terminal. Plain numbered input, not a TUI: arrow-key navigation would need raw
//! terminal mode, which needs either `unsafe` termios calls or a new dependency — neither
//! justified for a first-run convenience prompt the user can always skip with an explicit
//! `ralph run <model>`.
use std::io::{IsTerminal, Write};

/// A small curated starting point, not a registry: Ralph runs any vLLM-supported model id
/// via "enter another model...", or the existing `ralph run <model>` form.
const STARTER_MODELS: &[&str] = &[
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
];

pub fn should_prompt(json: bool) -> bool {
    !json && std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

fn read_line() -> Option<String> {
    let mut buf = String::new();
    match std::io::stdin().read_line(&mut buf) {
        Ok(0) => None, // Ctrl-D
        Ok(_) => Some(buf),
        Err(_) => None,
    }
}

/// `None` on Ctrl-D — Ctrl-C isn't handled specially here: nothing has been created yet
/// at this point, so the default terminate-the-process behavior is already safe.
pub fn pick_model() -> Option<String> {
    println!("pick a model:");
    for (i, m) in STARTER_MODELS.iter().enumerate() {
        println!("  {}) {m}", i + 1);
    }
    let other = STARTER_MODELS.len() + 1;
    println!("  {other}) enter another model...");

    loop {
        print!("> ");
        let _ = std::io::stdout().flush();
        let choice = read_line()?;
        let choice = choice.trim();

        if let Ok(n) = choice.parse::<usize>() {
            if (1..=STARTER_MODELS.len()).contains(&n) {
                return Some(STARTER_MODELS[n - 1].to_string());
            }
            if n == other {
                print!("model id: ");
                let _ = std::io::stdout().flush();
                let model = read_line()?;
                let model = model.trim();
                if !model.is_empty() {
                    return Some(model.to_string());
                }
                println!("model id cannot be empty");
                continue;
            }
        }
        println!("enter a number from 1 to {other}");
    }
}

/// Shows `default`, returning it unchanged if the user just presses Enter.
pub fn pick_session_name(default: &str) -> Option<String> {
    print!("session name [{default}]: ");
    let _ = std::io::stdout().flush();
    let line = read_line()?;
    let trimmed = line.trim();
    Some(if trimmed.is_empty() {
        default.to_string()
    } else {
        trimmed.to_string()
    })
}

/// A short, display-only suggestion — never stored as an alias, just a starting point the
/// user can freely overwrite in `pick_session_name`.
pub fn slugify(model: &str) -> String {
    let short = model.rsplit('/').next().unwrap_or(model);
    let mut slug = String::new();
    for c in short.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
        } else if !slug.ends_with('-') && !slug.is_empty() {
            slug.push('-');
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        "session".to_string()
    } else {
        slug
    }
}

/// `base`, or `base-2`, `base-3`, ... — the first suffix skips straight to 2 since `base`
/// itself stands in for "1".
pub fn unique_name(base: &str, existing: &[String]) -> String {
    if !existing.iter().any(|n| n == base) {
        return base.to_string();
    }
    let mut n = 2;
    loop {
        let candidate = format!("{base}-{n}");
        if !existing.iter().any(|name| name == &candidate) {
            return candidate;
        }
        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_strips_org_prefix_and_lowercases() {
        assert_eq!(slugify("Qwen/Qwen3-0.6B"), "qwen3-0-6b");
    }

    #[test]
    fn slugify_collapses_repeated_separators() {
        assert_eq!(slugify("Foo/Bar..Baz"), "bar-baz");
    }

    #[test]
    fn slugify_never_returns_empty() {
        assert_eq!(slugify("///"), "session");
    }

    #[test]
    fn unique_name_returns_base_when_free() {
        assert_eq!(unique_name("qwen3", &[]), "qwen3");
    }

    #[test]
    fn unique_name_skips_straight_to_dash_two() {
        let existing = vec!["qwen3".to_string()];
        assert_eq!(unique_name("qwen3", &existing), "qwen3-2");
    }

    #[test]
    fn unique_name_keeps_incrementing_past_taken_suffixes() {
        let existing = vec![
            "qwen3".to_string(),
            "qwen3-2".to_string(),
            "qwen3-3".to_string(),
        ];
        assert_eq!(unique_name("qwen3", &existing), "qwen3-4");
    }
}
