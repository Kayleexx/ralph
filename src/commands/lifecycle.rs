use super::*;

pub async fn run_checkpoint(home: &Path, cli: Flags, session: String) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    let request = Request::Checkpoint { session };
    match client::send_request(&mut stream, &request).await {
        Ok(Response::Run(info)) => {
            if cli.json {
                print_json(&info);
            } else if cli.quiet {
                println!("{}", info.name);
            } else {
                println!(
                    "{} {} checkpointed",
                    style(cli, "32", ok_mark(cli)),
                    style(cli, "1", &info.name)
                );
            }
            0
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}

pub async fn run_pause(home: &Path, cli: Flags, session: String) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    let request = Request::Pause { session };
    let result = if !cli.json && !cli.quiet && std::io::stdout().is_terminal() {
        send_with_progress(&mut stream, &request, "pausing").await
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
                    "{} {} paused",
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

pub async fn run_hibernate(home: &Path, cli: Flags, session: String) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    let request = Request::Hibernate { session };
    let result = if !cli.json && !cli.quiet && std::io::stdout().is_terminal() {
        send_with_progress(&mut stream, &request, "hibernating").await
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
                    "{} {} hibernated",
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

pub async fn run_resume(
    home: &Path,
    cli: Flags,
    session: String,
    fast_only: bool,
    portable: bool,
) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    let request = Request::Resume {
        session,
        fast_only,
        portable,
    };
    let result = if !cli.json && !cli.quiet && std::io::stdout().is_terminal() {
        send_with_progress(&mut stream, &request, "resuming").await
    } else {
        client::send_request(&mut stream, &request).await
    };
    match result {
        Ok(Response::Resume(info)) => {
            if cli.json {
                print_json(&info);
            } else if cli.quiet {
                println!("{}", info.session.name);
            } else if info.native {
                println!("native checkpoint compatible");
                println!(
                    "{} restored {}",
                    style(cli, "32", ok_mark(cli)),
                    style(cli, "1", &info.session.name)
                );
            } else {
                println!(
                    "native checkpoint unavailable; rehydrating from {} tokens...",
                    info.session.token_count
                );
                println!(
                    "{} restored {}",
                    style(cli, "32", ok_mark(cli)),
                    style(cli, "1", &info.session.name)
                );
            }
            0
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}
