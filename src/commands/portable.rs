use super::*;

/// Resolved against the CLI's own cwd — the daemon process has its own, unrelated cwd,
/// so a relative `--output`/path must never be sent to it unresolved.
fn resolve_path(path: &str) -> std::path::PathBuf {
    let path = std::path::Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    }
}

pub async fn run_export(
    home: &Path,
    cli: Flags,
    session: String,
    output: Option<String>,
    force: bool,
    with_accel: bool,
) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    let output_path = resolve_path(&output.unwrap_or_else(|| format!("{session}.ralph")));
    let request = Request::Export {
        session,
        output_path: output_path.to_string_lossy().into_owned(),
        force,
        with_accel,
    };
    let result = if !cli.json && !cli.quiet {
        send_with_progress(&mut stream, &request, "exporting").await
    } else {
        client::send_request(&mut stream, &request).await
    };
    match result {
        Ok(Response::Run(info)) => {
            if cli.json {
                print_json(&serde_json::json!({
                    "session": info,
                    "output": output_path.to_string_lossy(),
                }));
            } else if cli.quiet {
                println!("{}", output_path.display());
            } else {
                println!(
                    "{} {} exported to {}",
                    style(cli, "32", ok_mark(cli)),
                    style(cli, "1", &info.name),
                    output_path.display()
                );
            }
            0
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}

pub async fn run_import(home: &Path, cli: Flags, path: String, name: Option<String>) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    let path = resolve_path(&path);
    let request = Request::Import {
        path: path.to_string_lossy().into_owned(),
        name,
    };
    let result = if !cli.json && !cli.quiet {
        send_with_progress(&mut stream, &request, "importing").await
    } else {
        client::send_request(&mut stream, &request).await
    };
    match result {
        Ok(Response::Run(info)) => {
            if cli.json {
                print_json(&info);
            } else if cli.quiet {
                println!("{}", info.name);
            } else {
                println!(
                    "{} {} imported",
                    style(cli, "32", ok_mark(cli)),
                    style(cli, "1", &info.name)
                );
                println!("{} ralph resume {}", arrow(cli), info.name);
            }
            0
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}
