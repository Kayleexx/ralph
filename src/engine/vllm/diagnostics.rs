use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub(super) fn tail(path: &Path) -> String {
    let Ok(mut file) = File::open(path) else {
        return String::new();
    };
    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    if file
        .seek(SeekFrom::Start(len.saturating_sub(16_384)))
        .is_err()
    {
        return String::new();
    }
    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() {
        return String::new();
    }
    let text = String::from_utf8_lossy(&bytes);
    let text = text
        .rsplit_once("--- Ralph startup ")
        .map_or(text.as_ref(), |(_, current)| current);
    text.to_string()
}

pub(super) fn cause(tail: &str) -> Option<String> {
    let lines: Vec<_> = tail.lines().collect();
    lines
        .iter()
        .rev()
        .find(|line| {
            let lower = line.to_lowercase();
            !lower
                .replace(' ', "")
                .contains("enginecoreinitializationfailed")
                && (lower.contains("out of memory")
                    || lower.contains("free memory")
                    || lower.contains("kv cache")
                        && (lower.contains("needed")
                            || lower.contains("larger")
                            || lower.contains("insufficient"))
                    || lower.contains("error: unrecognized arguments")
                    || lower.contains("valueerror:")
                    || lower.contains("runtimeerror:"))
        })
        .map(|line| {
            let message = line
                .split_once("ValueError: ")
                .or_else(|| line.split_once("RuntimeError: "))
                .map_or(*line, |(_, message)| message);
            message
                .split_once("] ")
                .map_or(message, |(_, message)| message)
                .trim()
                .split(" Based on")
                .next()
                .unwrap_or(line)
                .split(" Decrease")
                .next()
                .unwrap_or(line)
                .trim_start_matches("ValueError: ")
                .trim_start_matches("RuntimeError: ")
                .chars()
                .take(600)
                .collect()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn originating_capacity_error_beats_enginecore_wrapper() {
        let log = "ValueError: KV cache needed 4.38 GiB; available 2.94 GiB\nRuntimeError: EngineCore initialization failed";
        assert!(cause(log).unwrap().contains("4.38"));
    }

    #[test]
    fn long_wrapper_traceback_preserves_originating_capacity_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("startup.log");
        let origin =
            "(EngineCore pid=1) ValueError: KV cache needed 4.38 GiB; available 2.99 GiB\n";
        let wrapper = "(APIServer pid=2) RuntimeError: Engine core initialization failed";
        std::fs::write(
            &path,
            format!("{origin}{}{wrapper}", "wrapper frame\n".repeat(125)),
        )
        .unwrap();
        let diagnostic = tail(&path);
        assert!(diagnostic.len() <= 16_384);
        assert_eq!(
            cause(&diagnostic).unwrap(),
            "KV cache needed 4.38 GiB; available 2.99 GiB"
        );
    }
}
