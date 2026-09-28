use super::*;

pub async fn run_query(
    home: &Path,
    cli: Flags,
    session: String,
    prompt_arg: Option<String>,
) -> i32 {
    let prompt = match prompt_arg.as_deref() {
        None | Some("-") => {
            let mut buf = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
                return print_io_error(cli, e);
            }
            buf
        }
        Some(p) => p.to_string(),
    };
    let stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    match client::run_query(stream, &session, &prompt, cli.json).await {
        Ok(client::QueryResult::Outcome(outcome)) => {
            if cli.json {
                print_json(
                    &serde_json::json!({ "session": session, "text": outcome.text, "tokens": outcome.token_count }),
                );
            } else {
                println!();
            }
            i32::from(outcome.cancelled)
        }
        Ok(client::QueryResult::Failed(payload)) => print_error(cli, &payload),
        Err(e) => print_io_error(cli, e),
    }
}

pub async fn run_recover(home: &Path, cli: Flags, session: String) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    let request = Request::Recover { session };
    let result = if !cli.json && !cli.quiet && std::io::stdout().is_terminal() {
        send_with_progress(&mut stream, &request, "recovering").await
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
                    "{} {} recovered",
                    style(cli, "32", ok_mark(cli)),
                    style(cli, "1", &info.name)
                );
                println!(
                    "  model: {}",
                    format_model_ref(&info.model, info.model_revision.as_deref())
                );
                println!("{} ralph query {} \"continue\"", arrow(cli), info.name);
            }
            0
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}
