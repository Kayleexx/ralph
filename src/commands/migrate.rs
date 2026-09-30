use super::*;

pub async fn run_migrate(home: &Path, cli: Flags, session: String, to: u32) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    let request = Request::Migrate { session, to };
    let result = if !cli.json && !cli.quiet && std::io::stdout().is_terminal() {
        send_with_progress(&mut stream, &request, "migrating").await
    } else {
        client::send_request(&mut stream, &request).await
    };
    match result {
        Ok(Response::Migrate(report)) => {
            if cli.json {
                print_json(&report);
            } else if cli.quiet {
                println!("{}", report.session.name);
            } else {
                println!("preparing gpu{}", report.destination_gpu);
                println!("moving session");
                println!("resuming");
                println!();
                println!(
                    "{} done \u{b7} {:.0} ms interruption",
                    ok_mark(cli),
                    report.interruption_ms
                );
                if cli.verbose {
                    println!(
                        "  source gpu{} -> destination gpu{}, {} native",
                        report.source_gpu,
                        report.destination_gpu,
                        if report.native { "" } else { "not" }
                    );
                    println!(
                        "  total {:.0} ms, replay/prefill {:.0} ms, {} tokens",
                        report.total_ms, report.replay_prefill_ms, report.token_count
                    );
                }
            }
            0
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}
