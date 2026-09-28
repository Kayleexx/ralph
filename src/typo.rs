//! Suggests the closest existing name for a likely typo (e.g. `dem` -> `demo`), so a
//! not-found error can be genuinely helpful without ever silently picking a session.

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = if ca == cb {
                prev
            } else {
                1 + prev.min(above).min(row[j])
            };
            prev = above;
        }
    }
    row[b.len()]
}

/// Only suggests within a small edit distance relative to length, so an unrelated name
/// never gets offered as a "did you mean" for a wildly different identifier.
pub fn suggest_similar<'a>(
    target: &str,
    candidates: impl Iterator<Item = &'a str>,
) -> Option<String> {
    let target_lower = target.to_lowercase();
    candidates
        .map(|c| (c, levenshtein(&target_lower, &c.to_lowercase())))
        .filter(|(c, dist)| *dist > 0 && *dist <= (c.len().max(target.len()) / 2).max(2))
        .min_by_key(|(_, dist)| *dist)
        .map(|(c, _)| c.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_typo_suggests_the_full_name() {
        assert_eq!(
            suggest_similar("dem", ["demo", "other"].into_iter()),
            Some("demo".to_string())
        );
    }

    #[test]
    fn exact_match_suggests_nothing() {
        assert_eq!(suggest_similar("demo", ["demo"].into_iter()), None);
    }

    #[test]
    fn unrelated_name_suggests_nothing() {
        assert_eq!(
            suggest_similar("xyz123", ["demo", "production"].into_iter()),
            None
        );
    }

    #[test]
    fn no_candidates_suggests_nothing() {
        assert_eq!(suggest_similar("dem", std::iter::empty()), None);
    }
}
