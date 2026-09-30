//! Read-only reporting: `ralph ps`/`inspect`/`doctor`.
use super::*;

pub async fn run_ps(home: &Path, cli: Flags) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    match client::send_request(&mut stream, &Request::Ps).await {
        Ok(Response::Ps(sessions)) => {
            if cli.json {
                print_json(&sessions);
            } else if cli.quiet {
                for s in &sessions {
                    println!("{}", s.name);
                }
            } else if sessions.is_empty() {
                println!(
                    "no sessions yet\n\n{} ralph run <model> --name <name>",
                    arrow(cli)
                );
            } else {
                let header = format!(
                    "{:<16} {:<24} {:>8} {:<10} LOCATION",
                    "NAME", "MODEL", "TOKENS", "STATE"
                );
                println!("{}", style(cli, "2", &header));
                for s in &sessions {
                    println!(
                        "{:<16} {:<24} {:>8} {:<10} {}",
                        s.name,
                        short_model_name(&s.model),
                        s.token_count,
                        s.state,
                        s.location
                    );
                    if cli.verbose {
                        println!(
                            "  id: {}  pid: {}",
                            s.id,
                            s.pid
                                .map(|p| p.to_string())
                                .unwrap_or_else(|| "-".to_string())
                        );
                    }
                }
            }
            0
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}

pub async fn run_inspect(home: &Path, cli: Flags, session: String) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    match client::send_request(&mut stream, &Request::Inspect { session }).await {
        Ok(Response::Inspect(info)) => {
            if cli.json {
                print_json(&info);
            } else if cli.quiet {
                println!("{}", info.session.state);
            } else {
                println!("{} ({})", info.session.name, info.session.state);
                println!(
                    "  model: {}",
                    format_model_ref(&info.session.model, info.session.model_revision.as_deref())
                );
                println!("  recoverability: {}", info.recoverability);
                println!("  resume: {}", info.restore_readiness);
                println!("  portable state: {}", info.portable_state);
                if info.continuity_policy != "warm" || info.continuity_target_ms.is_some() {
                    print!("  continuity: {}", info.continuity_policy);
                    if let Some(target) = info.continuity_target_ms {
                        print!(" (target {target} ms)");
                    }
                    println!();
                }
                if let Some(destination) = info.moved_to {
                    println!("  moved to: {destination}");
                }
                if let Some(failure) = info.last_failure {
                    println!("  last worker failure: {failure}");
                }
                if cli.verbose
                    && let Some(diff) = info.fingerprint_diff
                {
                    for (field, current, stored) in diff {
                        println!("  {field}: {current} (checkpoint had {stored})");
                    }
                }
            }
            0
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}

// Runs entirely in the CLI process rather than round-tripping through the daemon: doctor
// exists to diagnose why the daemon/environment *isn't* working, so it must not depend on
// the daemon being startable, and must not create the data directory as a side effect of
// connecting to one.
pub fn run_doctor(home: &Path, cli: Flags) -> i32 {
    let checks = doctor::run_checks(home);
    let any_fail = checks.iter().any(|c| c.status == "fail");
    if cli.json {
        print_json(&checks);
    } else if cli.quiet {
        println!("{}", if any_fail { "fail" } else { "pass" });
    } else {
        for check in &checks {
            println!(
                "{:<5} {:<16} {}",
                check.status.to_uppercase(),
                check.name,
                check.detail
            );
        }
    }
    if any_fail { 6 } else { 0 }
}
