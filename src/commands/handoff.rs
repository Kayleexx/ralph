use super::*;

pub async fn run_handoff(
    home: &Path,
    cli: Flags,
    session: String,
    destination: String,
    name: Option<String>,
) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    let request = Request::Handoff {
        session,
        destination: destination.clone(),
        name,
    };
    let result = if !cli.json && !cli.quiet {
        client::send_cancellable_with_progress(&mut stream, &request, "handing off").await
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
                    "{} {} moved to {}",
                    style(cli, "32", ok_mark(cli)),
                    style(cli, "1", &info.name),
                    destination
                );
                println!(
                    "{} on {destination}: ralph resume {}",
                    arrow(cli),
                    info.name
                );
            }
            0
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}

pub async fn run_drain(
    home: &Path,
    cli: Flags,
    location: String,
    to: Option<String>,
    yes: bool,
) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    let request = Request::Drain { location, to, yes };
    let result = if !cli.json && !cli.quiet {
        client::send_cancellable_with_progress(&mut stream, &request, "draining").await
    } else {
        client::send_request(&mut stream, &request).await
    };
    match result {
        Ok(Response::Drain(report)) => {
            if cli.json {
                print_json(&report);
            } else {
                println!("draining {}\n", report.location);
                for outcome in &report.sessions {
                    let mark = if outcome.ok {
                        style(cli, "32", ok_mark(cli))
                    } else {
                        style(cli, "31", "x")
                    };
                    print!("{} {} -> {}", mark, outcome.name, outcome.action);
                    if let Some(detail) = &outcome.detail {
                        print!("  ({detail})");
                    }
                    println!();
                }
                println!("\n{} sessions", report.sessions.len());
                if !report.executed {
                    println!("{} rerun with --yes to execute", arrow(cli));
                } else if report.all_safe {
                    println!("{} gpu0 safe to stop", style(cli, "32", ok_mark(cli)));
                } else {
                    println!(
                        "{} drain incomplete; some sessions were not moved",
                        style(cli, "31", "x")
                    );
                }
            }
            if report.executed && !report.all_safe {
                6
            } else {
                0
            }
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}
