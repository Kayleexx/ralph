//! Resolves a Hugging Face Hub model to a real commit sha, so Ralph never pretends the
//! repo id itself is a revision.
use reqwest::Client;

fn info_url(model: &str, revision: Option<&str>) -> String {
    match revision {
        Some(rev) => format!("https://huggingface.co/api/models/{model}/revision/{rev}"),
        None => format!("https://huggingface.co/api/models/{model}"),
    }
}

/// Best-effort: returns `None` on any network/parse failure rather than erroring the
/// caller — an unresolved revision is represented honestly by the caller, not treated as
/// a fatal startup failure.
pub async fn resolve_revision(
    client: &Client,
    model: &str,
    revision: Option<&str>,
) -> Option<String> {
    let resp = client.get(info_url(model, revision)).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let json: serde_json::Value = resp.json().await.ok()?;
    let sha = json.get("sha")?.as_str()?;
    Some(sha.chars().take(12).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_revision_url_has_no_revision_segment() {
        assert_eq!(
            info_url("Qwen/Qwen2.5-0.5B-Instruct", None),
            "https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct"
        );
    }

    #[test]
    fn explicit_revision_is_included_in_the_url() {
        assert_eq!(
            info_url("Qwen/Qwen2.5-0.5B-Instruct", Some("main")),
            "https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct/revision/main"
        );
    }
}
