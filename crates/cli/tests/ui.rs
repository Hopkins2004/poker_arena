//! End-to-end test of `arena ui`: page, defaults, plan preview, a run with progress, result.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Server {
    child: Child,
    addr: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

fn start() -> Server {
    let mut child = Command::new(env!("CARGO_BIN_EXE_arena"))
        .args(["ui", "--port", "0"])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let addr = line
        .split("http://")
        .nth(1)
        .and_then(|s| s.split('/').next())
        .unwrap_or_else(|| panic!("unexpected banner: {line}"))
        .to_string();
    Server { child, addr }
}

fn http(addr: &str, method: &str, path: &str, body: Option<&Value>) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    let body = body.map(|b| b.to_string()).unwrap_or_default();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let code = out[9..12].parse().unwrap();
    let body = out.split_once("\r\n\r\n").unwrap().1.to_string();
    (code, body)
}

fn json(addr: &str, method: &str, path: &str, body: Option<&Value>) -> (u16, Value) {
    let (c, b) = http(addr, method, path, body);
    (c, serde_json::from_str(&b).unwrap())
}

#[test]
fn ui_end_to_end() {
    let srv = start();
    let a = srv.addr.as_str();

    let (code, page) = http(a, "GET", "/", None);
    assert_eq!(code, 200);
    assert!(page.contains("<title>Poker Arena</title>"));

    let (_, d) = json(a, "GET", "/api/defaults", None);
    let mut cfg = d["config"].clone();
    assert_eq!(cfg["bots"].as_array().unwrap().len(), 7);

    let (code, plan) = json(a, "POST", "/api/plan", Some(&json!({ "config": cfg })));
    assert_eq!(code, 200, "{plan}");
    assert_eq!(plan["tables_per_round"], 7);
    assert_eq!(plan["total_hands"], 35_000);

    let mut bad = cfg.clone();
    bad["bots"] = json!([]);
    let (code, e) = json(a, "POST", "/api/plan", Some(&json!({ "config": bad })));
    assert_eq!(code, 400);
    assert!(e["error"].as_str().unwrap().contains("bots"));

    // A small run: 7 bots, 2 rounds, 200 hands per table, 10 tables per round.
    cfg["table"]["hands_per_table"] = json!(200);
    cfg["run"]["rounds"] = json!(2);
    cfg["matchmaking"]["min_tables"] = json!(10);
    let (code, r) = json(
        a,
        "POST",
        "/api/run",
        Some(&json!({ "config": cfg, "save": false })),
    );
    assert_eq!(code, 200, "{r}");
    let t0 = Instant::now();
    let status = loop {
        let (_, s) = json(a, "GET", "/api/status", None);
        if s["state"] != "running" {
            break s;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(120),
            "run did not finish"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(status["state"], "done", "{status}");
    assert_eq!(status["tables_total"], 20);
    assert_eq!(status["hands_total"], 4000);
    assert_eq!(status["hands_done"], 4000);
    let rows = status["result"]["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 7);
    assert!(rows
        .windows(2)
        .all(|w| w[0]["rating"].as_f64() >= w[1]["rating"].as_f64()));
    assert!(
        status["result"]["results_dir"].is_null(),
        "save=false must not write"
    );
}
