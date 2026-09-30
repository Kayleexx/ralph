use super::*;
use crate::daemon::tests::{FakeEngine, test_daemon};
use std::time::Duration;

async fn setup() -> (
    tempfile::TempDir,
    Arc<Daemon<FakeEngine>>,
    UnixStream,
    tokio::task::JoinHandle<std::io::Result<()>>,
) {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".into(), Some("demo".into()))
        .await
        .unwrap();
    let (client, server) = UnixStream::pair().unwrap();
    let d = daemon.clone();
    let task = tokio::spawn(handle_connection(d, server));
    (dir, daemon, client, task)
}

#[tokio::test]
async fn stream_error_flushes_partial_and_never_sends_success() {
    let (_dir, daemon, mut client, task) = setup().await;
    ipc::write_frame(
        &mut client,
        &ipc::Request::Query {
            session: "demo".into(),
            prompt: "stream-error".into(),
        },
    )
    .await
    .unwrap();
    assert!(matches!(
        ipc::read_frame::<_, ServerMessage>(&mut client)
            .await
            .unwrap(),
        Some(ServerMessage::Chunk(_))
    ));
    assert!(
        matches!(ipc::read_frame::<_,ServerMessage>(&mut client).await.unwrap(),Some(ServerMessage::Result(response)) if matches!(*response,Response::Error(_)))
    );
    task.await.unwrap().unwrap();
    let id = daemon.inspect("demo").unwrap().session.id;
    assert_eq!(daemon.replay_messages(&id).unwrap()[1].content, "hi");
}

#[tokio::test]
async fn failed_flush_does_not_advance_offset_or_send_done() {
    let (_dir, daemon, mut client, task) = setup().await;
    daemon.inject_storage_fault("CREATE TRIGGER fail_flush BEFORE UPDATE ON token_turns BEGIN SELECT RAISE(ABORT,'injected failure'); END");
    ipc::write_frame(
        &mut client,
        &ipc::Request::Query {
            session: "demo".into(),
            prompt: "flush-failure".into(),
        },
    )
    .await
    .unwrap();
    assert!(matches!(
        ipc::read_frame::<_, ServerMessage>(&mut client)
            .await
            .unwrap(),
        Some(ServerMessage::Chunk(_))
    ));
    assert!(
        matches!(ipc::read_frame::<_,ServerMessage>(&mut client).await.unwrap(),Some(ServerMessage::Result(response)) if matches!(*response,Response::Error(_)))
    );
    task.await.unwrap().unwrap();
    let id = daemon.inspect("demo").unwrap().session.id;
    assert_eq!(daemon.replay_messages(&id).unwrap().len(), 1);
    daemon.inject_storage_fault("DROP TRIGGER fail_flush");
    daemon.flush_assistant_turn(&id, 1, "hi", &[1; 32]).unwrap();
    daemon.flush_assistant_turn(&id, 1, "hi", &[1; 32]).unwrap();
    assert_eq!(daemon.replay_messages(&id).unwrap().len(), 2);
}

#[tokio::test]
async fn disconnect_cancels_and_flushes_before_releasing_session() {
    let (_dir, daemon, mut client, task) = setup().await;
    ipc::write_frame(
        &mut client,
        &ipc::Request::Query {
            session: "demo".into(),
            prompt: "wait".into(),
        },
    )
    .await
    .unwrap();
    assert!(matches!(
        ipc::read_frame::<_, ServerMessage>(&mut client)
            .await
            .unwrap(),
        Some(ServerMessage::Chunk(_))
    ));
    drop(client);
    tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let id = daemon.inspect("demo").unwrap().session.id;
    assert_eq!(daemon.replay_messages(&id).unwrap()[1].content, "hi");
    assert!(daemon.begin_query("demo", "next").await.is_ok());
}

#[tokio::test]
async fn disconnect_during_startup_cancels_and_converges_to_stopped() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let (mut client, server) = UnixStream::pair().unwrap();
    let d = daemon.clone();
    let task = tokio::spawn(handle_connection(d, server));
    ipc::write_frame(
        &mut client,
        &ipc::Request::Run {
            model: "model".into(),
            name: Some("cancelled".into()),
            gpu: None,
            policy: None,
            continuity_target_ms: None,
        },
    )
    .await
    .unwrap();
    drop(client);
    let _ = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        daemon.inspect("cancelled").unwrap().session.state,
        "stopped"
    );
}
