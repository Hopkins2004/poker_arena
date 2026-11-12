//! `arena ui`: a small local web UI to configure a run, watch its progress and see the
//! leaderboard.
//!
//! A deliberately minimal HTTP/1.1 server on `std::net` (no extra dependencies), bound to
//! localhost by default. One run at a time executes on a background thread; the page polls
//! `/api/status`. Endpoints:
//!
//! * `GET  /`              the page
//! * `GET  /api/defaults`  default config (the 6-max demo) and the bot kinds
//! * `POST /api/plan`      `{config}` → tables per round, total tables/hands (or an error)
//! * `POST /api/run`       `{config, save}` → starts a run (409 if one is running)
//! * `GET  /api/status`    progress, and the leaderboard once finished

use arena::config::ConfigFile;
use arena::leaderboard::{utc_timestamp, write_results_dir};
use arena::matchmaking::{effective_seats, make_schedule};
use arena::{run_with_progress, BotEntry, LeaderboardFile, RunProgress};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const PAGE: &str = include_str!("ui.html");
const DEFAULT_CONFIG: &str = include_str!("../../../configs/6max_nlh.toml");

#[derive(Default)]
struct UiState {
    job: Option<Job>,
    next_id: u64,
}

struct Job {
    id: u64,
    progress: RunProgress,
    rounds_total: usize,
    started: Instant,
    finished: Option<f64>,
    outcome: Option<Result<Value, String>>,
}

type Shared = Arc<Mutex<UiState>>;

/// Serves the UI until the process is killed. Prints the URL once bound.
pub fn serve(host: &str, port: u16) -> Result<(), String> {
    let listener =
        TcpListener::bind((host, port)).map_err(|e| format!("cannot bind {host}:{port}: {e}"))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    println!("arena ui listening on http://{addr}/  (Ctrl-C to stop)");
    std::io::stdout().flush().ok();
    let state: Shared = Arc::default();
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let state = state.clone();
        std::thread::spawn(move || {
            let _ = handle(stream, &state);
        });
    }
    Ok(())
}

fn handle(mut stream: TcpStream, state: &Shared) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, path) = (
        parts.next().unwrap_or("").to_string(),
        parts.next().unwrap_or("").to_string(),
    );
    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    if content_length > 1 << 20 {
        return respond(
            &mut stream,
            413,
            "application/json",
            r#"{"error":"body too large"}"#,
        );
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);

    let path = path.split('?').next().unwrap_or("");
    let (code, ctype, out) = match (method.as_str(), path) {
        ("GET", "/") => (200, "text/html; charset=utf-8", PAGE.to_string()),
        ("GET", "/api/defaults") => json_ok(defaults()),
        ("POST", "/api/plan") => match plan(&body["config"]) {
            Ok(v) => json_ok(v),
            Err(e) => json_err(400, e),
        },
        ("POST", "/api/run") => start(state, &body),
        ("GET", "/api/status") => json_ok(status(state)),
        _ => json_err(404, "not found".into()),
    };
    respond(&mut stream, code, ctype, &out)
}

fn json_ok(v: Value) -> (u16, &'static str, String) {
    (200, "application/json", v.to_string())
}

fn json_err(code: u16, e: String) -> (u16, &'static str, String) {
    (code, "application/json", json!({ "error": e }).to_string())
}

fn respond(s: &mut TcpStream, code: u16, ctype: &str, body: &str) -> std::io::Result<()> {
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        413 => "Payload Too Large",
        _ => "Error",
    };
    write!(
        s,
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    s.flush()
}

fn defaults() -> Value {
    let cfg = ConfigFile::parse(DEFAULT_CONFIG).expect("bundled config is valid");
    json!({
        "config": serde_json::to_value(&cfg).unwrap(),
        "bot_kinds": bots::KINDS,
        "bot_params": {
            "random": {"fold": 1.0, "call": 2.0, "raise": 1.0, "raise_sizes": [0.5, 1.0]},
            "calling_station": {},
            "tag": {"samples": 300, "tightness": 0.0, "call_margin": 0.05},
        },
    })
}

fn parse_config(v: &Value) -> Result<ConfigFile, String> {
    let mut cfg: ConfigFile =
        serde_json::from_value(v.clone()).map_err(|e| format!("invalid config: {e}"))?;
    for b in &mut cfg.bots {
        if b.params.is_null() {
            b.params = json!({});
        }
        if b.version.as_deref() == Some("") {
            b.version = None;
        }
    }
    cfg.run_spec()?;
    Ok(cfg)
}

fn plan(v: &Value) -> Result<Value, String> {
    let cfg = parse_config(v)?;
    let spec = cfg.run_spec()?;
    let k = cfg.bots.len();
    let max_seats = spec.format.max_seats as usize;
    let sched = make_schedule(k, max_seats, &spec.matchmaking, 0).map_err(|e| e.to_string())?;
    let tables = sched.tables.len();
    let seats = effective_seats(k, max_seats, &spec.matchmaking);
    let apps = sched.appearances(k);
    Ok(json!({
        "bots": k,
        "seats_per_table": seats,
        "tables_per_round": tables,
        "total_tables": tables * spec.rounds,
        "total_hands": (tables * spec.rounds) as u64 * spec.table.hands,
        "appearances": [apps.iter().min(), apps.iter().max()],
        "tournament": spec.table.is_tournament(),
    }))
}

fn start(state: &Shared, body: &Value) -> (u16, &'static str, String) {
    let cfg = match parse_config(&body["config"]) {
        Ok(c) => c,
        Err(e) => return json_err(400, e),
    };
    let save = body["save"].as_bool().unwrap_or(false);
    let spec = match cfg.run_spec() {
        Ok(s) => s,
        Err(e) => return json_err(400, e),
    };
    let mut entries = Vec::new();
    for b in &cfg.bots {
        if b.python_target().is_some() {
            return json_err(
                400,
                format!("bot `{}`: Python bots run through pokerarena.run", b.name),
            );
        }
        match bots::factory(&b.kind, &b.name, &b.params) {
            Ok(factory) => entries.push(BotEntry {
                name: b.name.clone(),
                version: b.version.clone(),
                factory,
            }),
            Err(e) => return json_err(400, format!("bot `{}`: {e}", b.name)),
        }
    }

    let mut st = state.lock().unwrap();
    if st.job.as_ref().is_some_and(|j| j.outcome.is_none()) {
        return json_err(409, "a run is already in progress".into());
    }
    st.next_id += 1;
    let progress = RunProgress::hidden();
    let id = st.next_id;
    st.job = Some(Job {
        id,
        progress: progress.clone(),
        rounds_total: spec.rounds,
        started: Instant::now(),
        finished: None,
        outcome: None,
    });
    drop(st);

    let state = state.clone();
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let outcome = execute(&cfg, &spec, &entries, save, &progress, t0);
        let mut st = state.lock().unwrap();
        if let Some(job) = st.job.as_mut().filter(|j| j.id == id) {
            job.finished = Some(t0.elapsed().as_secs_f64());
            job.outcome = Some(outcome);
        }
    });
    json_ok(json!({ "started": id }))
}

fn execute(
    cfg: &ConfigFile,
    spec: &arena::RunSpec,
    entries: &[BotEntry],
    save: bool,
    progress: &RunProgress,
    t0: Instant,
) -> Result<Value, String> {
    let keys: Vec<String> = entries.iter().map(|e| e.key()).collect();
    let lb_path = PathBuf::from(&cfg.run.leaderboard);
    let mut lb = if save {
        LeaderboardFile::load(&lb_path).map_err(|e| format!("{}: {e}", lb_path.display()))?
    } else {
        LeaderboardFile::default()
    };
    let initial = lb.initial_ratings(&keys, spec.rating.initial);
    let out = run_with_progress(spec, entries, initial, progress).map_err(|e| e.to_string())?;
    let elapsed = t0.elapsed().as_secs_f64();
    let rows = out.state.rows();
    let hands: u64 = out
        .rounds
        .iter()
        .flat_map(|r| &r.results)
        .map(|t| t.hands_played)
        .sum();
    let tables: usize = out.rounds.iter().map(|r| r.results.len()).sum();
    let mut results_dir = None;
    if save {
        let run_id = format!("{}-s{}", utc_timestamp(), spec.seed);
        let text =
            toml::to_string(cfg).unwrap_or_else(|e| format!("# could not render config: {e}\n"));
        let dir = write_results_dir(
            &PathBuf::from(&cfg.run.results_dir),
            &run_id,
            &format!("# generated by `arena ui`\n{text}"),
            &keys,
            &out.rounds,
            &rows,
        )
        .map_err(|e| format!("writing results: {e}"))?;
        lb.update(&rows, spec.table.reference_bb(), &run_id);
        lb.save(&lb_path)
            .map_err(|e| format!("{}: {e}", lb_path.display()))?;
        results_dir = Some(dir.display().to_string());
    }
    Ok(json!({
        "rows": rows,
        "hands": hands,
        "tables": tables,
        "elapsed": elapsed,
        "hands_per_sec": hands as f64 / elapsed.max(1e-9),
        "results_dir": results_dir,
        "leaderboard": save.then(|| lb_path.display().to_string()),
        "tournament": spec.table.is_tournament(),
    }))
}

fn status(state: &Shared) -> Value {
    let st = state.lock().unwrap();
    let Some(job) = &st.job else {
        return json!({ "state": "idle" });
    };
    let p = &job.progress;
    let elapsed = job
        .finished
        .unwrap_or_else(|| job.started.elapsed().as_secs_f64());
    let hands_done = p.hands_done.load(Ordering::Relaxed);
    let mut v = json!({
        "id": job.id,
        "state": match &job.outcome {
            None => "running",
            Some(Ok(_)) => "done",
            Some(Err(_)) => "error",
        },
        "hands_done": hands_done,
        "hands_total": p.hands_total.load(Ordering::Relaxed),
        "tables_done": p.tables_done.load(Ordering::Relaxed),
        "tables_total": p.tables_total.load(Ordering::Relaxed),
        "rounds_done": p.rounds_done.load(Ordering::Relaxed),
        "rounds_total": job.rounds_total,
        "elapsed": elapsed,
        "hands_per_sec": hands_done as f64 / elapsed.max(1e-9),
    });
    match &job.outcome {
        Some(Ok(r)) => v["result"] = r.clone(),
        Some(Err(e)) => v["error"] = json!(e),
        None => {}
    }
    v
}
