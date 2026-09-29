use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

pub fn vram() -> Value {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--id=0",
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let line = String::from_utf8(out.stdout).unwrap();
    let parts: Vec<_> = line.trim().split(',').map(str::trim).collect();
    let total = parts[1].parse::<u64>().unwrap();
    let free = parts[2].parse::<u64>().unwrap();
    let apps = std::process::Command::new("nvidia-smi")
        .args([
            "--id=0",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .unwrap();
    assert!(apps.status.success());
    let allocations: Vec<_> = String::from_utf8(apps.stdout)
        .unwrap()
        .lines()
        .map(|line| {
            let (pid, memory) = line.split_once(',').unwrap();
            let pid = pid.trim().parse::<u32>().unwrap();
            let group = std::fs::read_to_string(format!("/proc/{pid}/stat"))
                .ok()
                .and_then(|s| {
                    s.rsplit_once(')')
                        .and_then(|(_, r)| r.split_whitespace().nth(2))
                        .and_then(|s| s.parse::<u32>().ok())
                });
            json!({"pid":pid,"group":group,"mib":memory.trim().parse::<u64>().unwrap()})
        })
        .collect();
    json!({"gpu":parts[0],"total_mib":total,"free_mib":free,"occupied_mib":total-free,"allocations":allocations})
}

async fn generate(client: &reqwest::Client, worker: &Value, ids: &[u32]) -> Value {
    let start = Instant::now();
    let reply: Value = client.post(format!("{}/v1/completions", worker["endpoint"].as_str().unwrap()))
        .json(&json!({"model":worker["model"],"prompt":ids,"max_tokens":128,"temperature":0,"ignore_eos":true}))
        .send().await.unwrap().error_for_status().unwrap().json().await.unwrap();
    assert_eq!(reply["usage"]["prompt_tokens"], 256);
    assert_eq!(reply["usage"]["completion_tokens"], 128);
    let seconds = start.elapsed().as_secs_f64();
    json!({"model":worker["model"],"latency_s":seconds,"output_tokens":128,"tokens_per_s":128.0/seconds})
}

pub async fn measure(workers: &[Value]) -> Value {
    let client = reqwest::Client::new();
    let mut prompts = Vec::new();
    for worker in workers {
        let tokens: Value = client.post(format!("{}/tokenize",worker["endpoint"].as_str().unwrap()))
            .json(&json!({"model":worker["model"],"prompt":"Explain how a lighthouse guides ships and how its light works. "}))
            .send().await.unwrap().error_for_status().unwrap().json().await.unwrap();
        let pattern: Vec<u32> = serde_json::from_value(tokens["tokens"].clone()).unwrap();
        assert!(!pattern.is_empty());
        let ids: Vec<_> = pattern.into_iter().cycle().take(256).collect();
        generate(&client, worker, &ids).await;
        prompts.push(ids);
    }
    let running = Arc::new(AtomicBool::new(true));
    let flag = running.clone();
    let sampler = tokio::spawn(async move {
        let mut minimum = u64::MAX;
        while flag.load(Ordering::SeqCst) {
            minimum = minimum.min(vram()["free_mib"].as_u64().unwrap());
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        minimum
    });
    let mut sequential = Vec::new();
    for (worker, ids) in workers.iter().zip(&prompts) {
        for _ in 0..3 {
            sequential.push(generate(&client, worker, ids).await);
        }
    }
    let mut concurrent = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let (a, b) = tokio::join!(
            generate(&client, &workers[0], &prompts[0]),
            generate(&client, &workers[1], &prompts[1])
        );
        concurrent.push(
            json!({"runs":[a,b],"aggregate_tokens_per_s":256.0/start.elapsed().as_secs_f64()}),
        );
    }
    running.store(false, Ordering::SeqCst);
    let minimum = sampler.await.unwrap();
    assert!(minimum >= 1024, "headroom exhausted: {minimum}");
    json!({"workers":workers,"vram":vram(),"minimum_sampled_free_mib":minimum,"sequential":sequential,"concurrent":concurrent})
}
