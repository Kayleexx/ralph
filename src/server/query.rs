use super::*;

const FLUSH_TOKEN_BATCH: usize = 32;
const FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// If the session needs recovering or resuming before it can serve a query, does that
/// now — streaming a `Status` frame before starting and another once it's ready — and
/// otherwise returns immediately. Shared by `handle_query` (every query, so `ralph
/// query`/`ralph chat` never need a manual `recover`/`resume` in between) and the
/// `EnsureReady` request (`ralph chat`'s startup, so the first prompt a user types
/// isn't also paying for recovery latency).
pub(super) async fn ensure_ready<E: Engine + 'static>(
    daemon: &Arc<Daemon<E>>,
    stream: &mut UnixStream,
    session: &str,
) -> Result<(), CliError> {
    let (row, action) = daemon.readiness_action(session)?;
    let Some(action) = action else {
        return Ok(());
    };
    let (mut reader, mut writer) = stream.split();
    ipc::write_frame(
        &mut writer,
        &ServerMessage::Status(format!("{} {}", action.label(), row.name)),
    )
    .await
    .map_err(|e| CliError::Other(e.into()))?;
    let start = std::time::Instant::now();
    let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
    let op = daemon.ready_up(&row.id, action, Some(cancel_rx));
    tokio::pin!(op);
    let mut outcome = tokio::select! {
        result = &mut op => result,
        _ = ipc::read_frame::<_, Cancel>(&mut reader) => {
            let _ = cancel_tx.send(());
            op.await
        }
    };
    // The crash supervisor may already be restarting this worker in the background —
    // don't retry the lock ourselves (a third contender), just watch state settle.
    if outcome.is_err() {
        for _ in 0..100 {
            let Ok((row, action)) = daemon.readiness_action(&row.id) else {
                break;
            };
            if row.state == crate::state::SessionState::Active {
                outcome = Ok(());
                break;
            }
            if action.is_none() {
                break;
            }
            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_millis(300)) => {}
                _ = ipc::read_frame::<_, Cancel>(&mut reader) => break,
            }
        }
    }
    outcome?;
    ipc::write_frame(
        &mut writer,
        &ServerMessage::Status(format!(
            "ready \u{b7} {:.1}s",
            start.elapsed().as_secs_f64()
        )),
    )
    .await
    .map_err(|e| CliError::Other(e.into()))?;
    Ok(())
}

pub(super) async fn handle_query<E: Engine + 'static>(
    daemon: &Arc<Daemon<E>>,
    stream: &mut UnixStream,
    session: &str,
    prompt: &str,
) -> std::io::Result<()> {
    if let Err(e) = ensure_ready(daemon, stream, session).await {
        return send_result(stream, Err(e)).await;
    }
    let running = match daemon.begin_query(session, prompt).await {
        Ok(r) => r,
        Err(e) => return send_result(stream, Err(e)).await,
    };
    let seq = match daemon.record_user_turn(&running.session_id, prompt, &running.input_ids) {
        Ok(s) => s,
        Err(e) => return send_result(stream, Err(e)).await,
    };
    let generation = running
        .engine
        .lock()
        .await
        .generate(&running.messages)
        .await;
    let handle = match generation {
        Ok(h) => h,
        Err(e) => return send_result(stream, Err(map_engine_error(e))).await,
    };
    let (mut reader, mut writer) = stream.split();
    let control = ipc::read_frame::<_, Cancel>(&mut reader);
    tokio::pin!(control);
    let mut tokens = handle.tokens;
    let mut cancel = Some(handle.cancel);
    let mut content = String::new();
    let mut ids = Vec::new();
    let mut committed = (0, 0);
    let mut connected = true;
    let mut completed = false;
    let mut flush_timer = tokio::time::interval(FLUSH_INTERVAL);
    flush_timer.tick().await;
    let mut result: Result<(), CliError> = loop {
        tokio::select! {
            chunk = tokens.recv() => match chunk {
                Some(Ok(chunk)) => {
                    content.push_str(&chunk.text);
                    ids.extend(chunk.ids);
                    if let Err(error) = ipc::write_frame(&mut writer, &ServerMessage::Chunk(chunk.text)).await {
                        connected = false;
                        break Err(CliError::Other(error.into()));
                    }
                    if ids.len() - committed.1 >= FLUSH_TOKEN_BATCH
                        && let Err(error) = flush(daemon, &running.session_id, seq, &content, &ids, &mut committed) { break Err(error); }
                }
                Some(Err(error)) => break Err(map_engine_error(error)),
                None => { completed = true; break Ok(()); }
            },
            _ = flush_timer.tick() => {
                if let Err(error) = flush(daemon, &running.session_id, seq, &content, &ids, &mut committed) { break Err(error); }
            }
            control = &mut control => { connected = matches!(control, Ok(Some(_))); break Ok(()); }
        }
    };
    if !completed && let Some(tx) = cancel.take() {
        // A closed receiver means generation has already ended.
        let _ = tx.send(());
    }
    // Retry cumulative output on every exit; advance the offset only after commit.
    if let Err(error) = flush(
        daemon,
        &running.session_id,
        seq,
        &content,
        &ids,
        &mut committed,
    ) {
        result = Err(error);
    }
    if let Err(error) =
        daemon.finish_generation(&running.session_id, seq, completed && result.is_ok())
    {
        result = Err(error);
    }
    if !connected {
        return result.map_err(std::io::Error::other);
    }
    match result {
        Ok(()) => {
            ipc::write_frame(
                &mut writer,
                &ServerMessage::Done {
                    token_count: ids.len() as i64,
                },
            )
            .await
        }
        Err(error) => send_result(&mut writer, Err(error)).await,
    }
}

fn flush<E: Engine + 'static>(
    daemon: &Daemon<E>,
    session: &str,
    seq: i64,
    content: &str,
    ids: &[u32],
    committed: &mut (usize, usize),
) -> Result<(), CliError> {
    if *committed != (content.len(), ids.len()) {
        daemon.flush_assistant_turn(session, seq, content, ids)?;
        *committed = (content.len(), ids.len());
    }
    Ok(())
}
