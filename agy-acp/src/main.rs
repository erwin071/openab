use fs2::FileExt;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    id: Option<Value>,
    method: Option<String>,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcNotification {
    jsonrpc: &'static str,
    method: String,
    params: Value,
}

/// Persisted session→conversation mapping stored in ~/.openab/agy-acp/sessions.json
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SessionStore {
    sessions: HashMap<String, StoredSession>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredSession {
    conversation_id: Option<String>,
    /// Last step idx read from SQLite; used for delta extraction.
    #[serde(default)]
    last_step_idx: i64,
    /// Selected model ID for this session.
    #[serde(default)]
    model_id: Option<String>,
}

struct Session {
    conversation_id: Option<String>,
    /// Last step idx read from SQLite.
    last_step_idx: i64,
    /// Selected model ID for this session.
    model_id: Option<String>,
}

/// Tracks streaming poll state shared between the polling thread and main task.
struct StreamingState {
    conversation_id: Option<String>,
    base_step_idx: i64,
    last_step_idx: i64,
    /// Tracks how many bytes of text we've already emitted per step idx.
    emitted_len: HashMap<i64, usize>,
    /// Tracks which tool steps we've already emitted notifications for.
    emitted_tool_steps: HashSet<i64>,
    had_updates: bool,
}

struct Adapter {
    sessions: HashMap<String, Session>,
    working_dir: String,
    conversations_dir: PathBuf,
    state_file: PathBuf,
    available_models: Option<Vec<String>>,
}

impl Adapter {
    fn new() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let state_dir = PathBuf::from(&home).join(".openab/agy-acp");
        Self {
            sessions: HashMap::new(),
            working_dir: std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| "/tmp".to_string()),
            conversations_dir: PathBuf::from(&home).join(".gemini/antigravity-cli/conversations"),
            state_file: state_dir.join("sessions.json"),
            available_models: None,
        }
    }

    /// Cache file path for model list.
    fn models_cache_path(&self) -> PathBuf {
        self.state_file.with_file_name("models_cache.json")
    }

    /// Load cached models (any age — cache is never deleted).
    fn load_cached_models(&self) -> Option<Vec<String>> {
        let path = self.models_cache_path();
        let content = fs::read_to_string(&path).ok()?;
        serde_json::from_str::<Vec<String>>(&content).ok().filter(|v| !v.is_empty())
    }

    /// Persist model list to cache file.
    fn save_models_cache(&self, models: &[String]) {
        if let Some(parent) = self.models_cache_path().parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string(models) {
            let tmp = self.models_cache_path().with_extension("tmp");
            if fs::write(&tmp, &json).is_ok() {
                let _ = fs::rename(&tmp, self.models_cache_path());
            }
        }
    }

    /// Hardcoded static fallback (last resort).
    fn static_fallback_models() -> Vec<String> {
        vec![
            "gemini-2.5-pro".to_string(),
            "gemini-2.5-flash".to_string(),
            "gemini-2.0-flash".to_string(),
        ]
    }

    /// Run `agy models` with 5s timeout and parse the output.
    fn fetch_available_models() -> Vec<String> {
        use std::time::Instant;
        let start = Instant::now();
        let mut child = match std::process::Command::new("agy")
            .arg("models")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() { return Vec::new(); }
                    let stdout = child.stdout.take().unwrap();
                    use std::io::Read;
                    let mut buf = String::new();
                    let _ = std::io::BufReader::new(stdout).read_to_string(&mut buf);
                    return buf.lines()
                        .map(|l| l.trim().to_string())
                        .filter(|l| !l.is_empty())
                        .collect();
                }
                Ok(None) => {
                    if start.elapsed() > Duration::from_secs(5) {
                        let _ = child.kill();
                        return Vec::new();
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(_) => return Vec::new(),
            }
        }
    }

    /// Get available models — always attempts fresh fetch (5s timeout):
    /// 1. Fetch `agy models` → success → use + always update cache
    /// 2. Fetch failed → cache (any age, never deleted) → use
    /// 3. No cache → static hardcoded list (last resort)
    fn get_available_models(&mut self) -> &[String] {
        if self.available_models.is_none() {
            let models = Self::fetch_available_models();
            if !models.is_empty() {
                eprintln!("[agy-acp] fetched {} models from `agy models`, updating cache", models.len());
                self.save_models_cache(&models);
                self.available_models = Some(models);
            } else if let Some(cached) = self.load_cached_models() {
                eprintln!("[agy-acp] `agy models` failed, using cached model list ({} models)", cached.len());
                self.available_models = Some(cached);
            } else {
                eprintln!("[agy-acp] `agy models` failed and no cache found, using hardcoded fallback");
                self.available_models = Some(Self::static_fallback_models());
            }
        }
        self.available_models.as_ref().unwrap()
    }

    /// Build the ACP configOptions JSON for the model selector.
    fn config_options_json(&mut self, model_id: Option<&str>) -> Value {
        let models = self.get_available_models();
        if models.is_empty() {
            return json!([]);
        }
        let current = model_id
            .or_else(|| models.first().map(|s| s.as_str()))
            .unwrap_or("");
        let options: Vec<Value> = models
            .iter()
            .map(|name| json!({ "value": name, "name": name }))
            .collect();
        json!([{
            "id": "model",
            "name": "Model",
            "category": "model",
            "type": "select",
            "currentValue": current,
            "options": options,
        }])
    }

    /// Acquire exclusive lock on a dedicated lock file for read-write mutual exclusion.
    fn lock_state_file(&self) -> Option<fs::File> {
        if let Some(parent) = self.state_file.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let lock_path = self.state_file.with_extension("lock");
        let lock_file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .ok()?;
        lock_file.lock_exclusive().ok()?;
        Some(lock_file)
    }

    /// Load persisted session store (caller must hold lock).
    fn load_store_inner(&self) -> SessionStore {
        let Some(file) = fs::File::open(&self.state_file).ok() else {
            return SessionStore::default();
        };
        serde_json::from_reader(&file).unwrap_or_default()
    }

    /// Load persisted session store with lock.
    fn load_store(&self) -> SessionStore {
        let _lock = self.lock_state_file();
        self.load_store_inner()
    }

    /// Try to restore conversation_id, last_step_idx, and model_id from persisted state.
    fn restore_session(&self, session_id: &str) -> Option<(String, i64, Option<String>)> {
        let store = self.load_store();
        store.sessions.get(session_id).and_then(|s| {
            s.conversation_id.clone().map(|cid| (cid, s.last_step_idx, s.model_id.clone()))
        })
    }

    /// Persist a session binding (read-modify-write under single lock).
    fn persist_session(&self, session_id: &str, conversation_id: Option<&str>, last_step_idx: i64, model_id: Option<&str>) {
        let Some(_lock) = self.lock_state_file() else {
            return;
        };
        let mut store = self.load_store_inner();
        store.sessions.insert(
            session_id.to_string(),
            StoredSession {
                conversation_id: conversation_id.map(String::from),
                last_step_idx,
                model_id: model_id.map(String::from),
            },
        );
        let tmp = self.state_file.with_extension("tmp");
        if let Ok(file) = fs::File::create(&tmp) {
            if serde_json::to_writer_pretty(&file, &store).is_ok() {
                let _ = fs::rename(&tmp, &self.state_file);
            }
        }
    }

    fn conversation_snapshot(&self) -> HashSet<String> {
        let Ok(entries) = fs::read_dir(&self.conversations_dir) else {
            return HashSet::new();
        };
        entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let path = e.path();
                if path.extension().map(|x| x == "db").unwrap_or(false) {
                    path.file_stem().map(|s| s.to_string_lossy().to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    fn new_conversation_id(&self, before: &HashSet<String>) -> Option<String> {
        let after = self.conversation_snapshot();
        let mut created: Vec<_> = after.difference(before).collect();
        if created.is_empty() {
            return None;
        }
        if created.len() > 1 {
            eprintln!(
                "[agy-acp] WARN: multiple new agy conversation files appeared; \
                 refusing to bind"
            );
            return None;
        }
        Some(created.remove(0).clone())
    }

    /// Extract text from a step_payload protobuf: top-level field 20 (sub-message) → field 1 (string).
    fn extract_text_from_step_payload(blob: &[u8]) -> Option<String> {
        let field_20 = Self::get_proto_field(blob, 20)?;
        let field_1 = Self::get_proto_field(&field_20, 1)?;
        String::from_utf8(field_1).ok()
    }

    /// Extract the first length-delimited field with the given number from a protobuf blob.
    fn get_proto_field(blob: &[u8], target: u64) -> Option<Vec<u8>> {
        let mut i = 0;
        while i < blob.len() {
            let (tag, consumed) = Self::read_varint(&blob[i..])?;
            i += consumed;
            let field_number = tag >> 3;
            let wire_type = tag & 0x7;
            match wire_type {
                0 => { let (_, c) = Self::read_varint(&blob[i..])?; i += c; }
                2 => {
                    let (len, c) = Self::read_varint(&blob[i..])?;
                    i += c;
                    let len = len as usize;
                    if i + len > blob.len() { return None; }
                    if field_number == target {
                        return Some(blob[i..i + len].to_vec());
                    }
                    i += len;
                }
                5 => { i += 4; }
                1 => { i += 8; }
                _ => return None,
            }
        }
        None
    }

    /// Read a protobuf varint, returning (value, bytes_consumed).
    fn read_varint(buf: &[u8]) -> Option<(u64, usize)> {
        let mut result: u64 = 0;
        let mut shift = 0;
        for (i, &byte) in buf.iter().enumerate() {
            if shift >= 70 {
                return None;
            }
            result |= ((byte & 0x7F) as u64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                return Some((result, i + 1));
            }
        }
        None
    }

    /// Get a text (UTF-8 string) from a protobuf field.
    fn get_proto_text(blob: &[u8], target: u64) -> Option<String> {
        let bytes = Self::get_proto_field(blob, target)?;
        String::from_utf8(bytes).ok()
    }

    /// Check if a step_type represents a tool call.
    fn is_tool_step_type(step_type: i64) -> bool {
        matches!(step_type, 5 | 7 | 8 | 9 | 17 | 21 | 33 | 101 | 138)
    }

    /// Extract tool name and input from a tool step payload.
    /// Parses: field 5 (tool sub-message) → field 4 (call) → field 2 or 9 (name), field 3 (input JSON).
    fn extract_tool_from_step_payload(blob: &[u8]) -> Option<(String, Option<Value>)> {
        let tool = Self::get_proto_field(blob, 5)?;
        let call = Self::get_proto_field(&tool, 4)?;
        let name = Self::get_proto_text(&call, 2)
            .or_else(|| Self::get_proto_text(&call, 9))
            .filter(|n| !n.is_empty())?;
        let input = Self::get_proto_text(&call, 3)
            .and_then(|s| serde_json::from_str::<Value>(&s).ok());
        Some((name, input))
    }

    /// Derive a short title for a tool call based on name and input.
    fn tool_call_title(name: &str, input: &Option<Value>) -> String {
        if let Some(input) = input {
            // Try common path fields
            for key in ["path", "file", "AbsolutePath", "FilePath"] {
                if let Some(path) = input.get(key).and_then(|v| v.as_str()) {
                    return format!("{}: {}", name, path);
                }
            }
            // Try query/command fields
            for key in ["query", "command", "text"] {
                if let Some(val) = input.get(key).and_then(|v| v.as_str()) {
                    let truncated: String = val.chars().take(60).collect();
                    return format!("{}: {}", name, truncated);
                }
            }
        }
        name.to_string()
    }

    /// Check if narration should be shown.
    /// - OPENAB_SHOW_NARRATION=1 → show (opt-in)
    /// - OPENAB_TOOL_DISPLAY=full → show (backward compat)
    /// - Default (neither set) → skip narration
    fn show_narration() -> bool {
        if let Ok(v) = std::env::var("OPENAB_SHOW_NARRATION") {
            return v == "1" || v.to_lowercase() == "true";
        }
        // Backward compat: OPENAB_TOOL_DISPLAY=full means show everything
        if let Ok(v) = std::env::var("OPENAB_TOOL_DISPLAY") {
            return v.to_lowercase() == "full";
        }
        false
    }

    /// Read the latest response from the SQLite conversation DB.
    /// Returns (response_text, max_step_idx) or None if reading fails.
    fn read_response_from_db(&self, conversation_id: &str, after_step_idx: i64) -> Option<(String, i64)> {
        let db_path = self.conversations_dir.join(format!("{}.db", conversation_id));
        let conn = Connection::open_with_flags(
            &db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        ).ok()?;

        // Verify steps table exists
        let table_exists: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='steps'",
            [],
            |row| row.get(0),
        ).unwrap_or(false);
        if !table_exists {
            eprintln!("[agy-acp] WARN: steps table not found in {}.db — schema changed?", conversation_id);
            return None;
        }

        let mut stmt = conn.prepare(
            "SELECT idx, step_payload FROM steps WHERE idx > ?1 AND step_type = 15 ORDER BY idx"
        ).ok()?;
        let rows: Vec<(i64, Vec<u8>)> = stmt.query_map([after_step_idx], |row| {
            Ok((row.get(0)?, row.get(1)?))
        }).ok()?.filter_map(|r| r.ok()).collect();

        let mut max_idx = after_step_idx;
        let mut response_parts: Vec<String> = Vec::new();
        for (idx, payload) in &rows {
            max_idx = max_idx.max(*idx);
            if let Some(text) = Self::extract_text_from_step_payload(payload) {
                if !text.is_empty() {
                    response_parts.push(text);
                }
            }
        }
        if response_parts.is_empty() {
            if !rows.is_empty() {
                let payload_sizes: Vec<usize> = rows.iter().map(|(_, p)| p.len()).collect();
                eprintln!(
                    "[agy-acp] WARN: {} new steps found (payload sizes: {:?}) but none had extractable text \
                     (field 20.1 missing — schema change?)",
                    rows.len(), payload_sizes
                );
            }
            return None;
        }
        let filtered = Self::filter_narration(&response_parts);
        Some((filtered, max_idx))
    }

    /// Poll the SQLite DB for new text since `base_step_idx` and emit streaming deltas.
    /// Returns notification JSON lines to write to stdout.
    fn poll_streaming_delta(
        conversations_dir: &PathBuf,
        snapshot: Option<&HashSet<String>>,
        session_id: &str,
        state: &Arc<Mutex<StreamingState>>,
    ) -> Vec<String> {
        // Try to bind conversation_id if not yet bound
        {
            let mut guard = state.lock().unwrap();
            if guard.conversation_id.is_none() {
                if let Some(before) = snapshot {
                    let after: HashSet<String> = fs::read_dir(conversations_dir)
                        .ok()
                        .map(|entries| {
                            entries
                                .filter_map(|e| e.ok())
                                .filter_map(|e| {
                                    let path = e.path();
                                    if path.extension().map(|x| x == "db").unwrap_or(false) {
                                        path.file_stem().map(|s| s.to_string_lossy().to_string())
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let mut created: Vec<_> = after.difference(before).collect();
                    if created.len() == 1 {
                        guard.conversation_id = Some(created.remove(0).clone());
                    }
                }
            }
        }

        let (conversation_id, base_step_idx) = {
            let guard = state.lock().unwrap();
            (guard.conversation_id.clone(), guard.base_step_idx)
        };

        let Some(conversation_id) = conversation_id else {
            return Vec::new();
        };

        // Read new rows from DB
        let db_path = conversations_dir.join(format!("{}.db", conversation_id));
        let conn = match Connection::open_with_flags(
            &db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        ) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut stmt = match conn.prepare(
            "SELECT idx, step_type, step_payload FROM steps WHERE idx > ?1 AND (step_type = 15 OR step_type IN (5,7,8,9,17,21,33,101,138)) ORDER BY idx"
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows: Vec<(i64, i64, Vec<u8>)> = stmt
            .query_map([base_step_idx], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .ok()
            .map(|iter| iter.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();

        let mut guard = state.lock().unwrap();
        let mut notifications = Vec::new();

        for (idx, step_type, payload) in rows {
            guard.last_step_idx = guard.last_step_idx.max(idx);

            if step_type == 15 {
                // Text response
                let Some(text) = Self::extract_text_from_step_payload(&payload) else {
                    continue;
                };

                let emitted = guard.emitted_len.get(&idx).copied().unwrap_or(0);
                if text.len() <= emitted {
                    continue;
                }

                // Skip narration by default (unless OPENAB_SHOW_NARRATION=1)
                if !Self::show_narration() && Self::is_narration(&text) {
                    guard.emitted_len.insert(idx, text.len());
                    continue;
                }

                let new_text = &text[emitted..];
                guard.emitted_len.insert(idx, text.len());

                if !new_text.is_empty() {
                    notifications.push(
                        serde_json::to_string(&JsonRpcNotification {
                            jsonrpc: "2.0",
                            method: "session/update".to_string(),
                            params: json!({
                                "sessionId": session_id,
                                "update": {
                                    "sessionUpdate": "agent_message_chunk",
                                    "content": { "type": "text", "text": new_text },
                                },
                            }),
                        })
                        .unwrap(),
                    );
                }
            } else if Self::is_tool_step_type(step_type) && !guard.emitted_tool_steps.contains(&idx) {
                // Tool call
                if let Some((name, input)) = Self::extract_tool_from_step_payload(&payload) {
                    guard.emitted_tool_steps.insert(idx);
                    let title = Self::tool_call_title(&name, &input);
                    let tool_call_id = format!("agy-{}-{}", idx, step_type);

                    // Emit tool_call (start)
                    let mut start_update = json!({
                        "sessionUpdate": "tool_call",
                        "toolCallId": tool_call_id,
                        "title": title,
                    });
                    if let Some(input) = &input {
                        start_update["rawInput"] = input.clone();
                    }
                    notifications.push(
                        serde_json::to_string(&JsonRpcNotification {
                            jsonrpc: "2.0",
                            method: "session/update".to_string(),
                            params: json!({
                                "sessionId": session_id,
                                "update": start_update,
                            }),
                        })
                        .unwrap(),
                    );

                    // Emit tool_call_update (completed)
                    notifications.push(
                        serde_json::to_string(&JsonRpcNotification {
                            jsonrpc: "2.0",
                            method: "session/update".to_string(),
                            params: json!({
                                "sessionId": session_id,
                                "update": {
                                    "sessionUpdate": "tool_call_update",
                                    "toolCallId": tool_call_id,
                                    "title": title,
                                    "status": "completed",
                                },
                            }),
                        })
                        .unwrap(),
                    );
                }
            }
        }

        guard.had_updates = guard.had_updates || !notifications.is_empty();
        notifications
    }

    /// Filter out leading narration ("I will ...") from response parts.
    /// Narration is skipped by default; set OPENAB_SHOW_NARRATION=1 to keep it.
    fn filter_narration(parts: &[String]) -> String {
        if Self::show_narration() || parts.len() <= 1 {
            return parts.join("\n");
        }

        let first_content = parts.iter().position(|p| !Self::is_narration(p)).unwrap_or(parts.len() - 1);
        parts[first_content..].join("\n")
    }

    /// A part is considered narration if every non-empty line starts with "I will".
    fn is_narration(text: &str) -> bool {
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return false;
        }
        lines.iter().all(|l| l.trim_start().starts_with("I will"))
    }

    fn evict_if_needed(&mut self) {
        const MAX_SESSIONS: usize = 64;
        while self.sessions.len() >= MAX_SESSIONS {
            if let Some(key) = self.sessions.keys().next().cloned() {
                self.sessions.remove(&key);
            }
        }
    }

    fn restore_session_state(&mut self, session_id: &str) -> bool {
        let Some((conversation_id, last_step_idx, model_id)) = self.restore_session(session_id) else {
            return false;
        };
        if !self.sessions.contains_key(session_id) {
            self.evict_if_needed();
        }
        self.sessions.insert(
            session_id.to_string(),
            Session {
                conversation_id: Some(conversation_id),
                last_step_idx,
                model_id,
            },
        );
        true
    }

    fn handle_initialize(&self, id: Value) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": 1,
                "agentInfo": { "name": "agy", "version": env!("CARGO_PKG_VERSION") },
                "agentCapabilities": { "streaming": true, "loadSession": true },
            })),
            error: None,
        }
    }

    fn handle_session_new(&mut self, id: Value) -> JsonRpcResponse {
        let session_id = Uuid::new_v4().to_string();
        self.evict_if_needed();
        self.sessions.insert(
            session_id.clone(),
            Session {
                conversation_id: None,
                last_step_idx: -1,
                model_id: None,
            },
        );
        let config_options = self.config_options_json(None);
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "sessionId": session_id, "configOptions": config_options })),
            error: None,
        }
    }

    fn handle_session_load(&mut self, id: Value, params: &Value) -> JsonRpcResponse {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if session_id.is_empty() {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32602,"message":"missing sessionId"})),
            };
        }

        if self.restore_session_state(session_id) {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({ "sessionId": session_id })),
                error: None,
            };
        }

        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(json!({
                "code": -32000,
                "message": format!("unknown sessionId: {session_id}"),
            })),
        }
    }

    fn handle_session_set_config_option(&mut self, id: Value, params: &Value) -> JsonRpcResponse {
        let session_id = params.get("sessionId").and_then(|v| v.as_str()).unwrap_or("");
        let config_id = params.get("configId").and_then(|v| v.as_str()).unwrap_or("");
        let value = params.get("value").and_then(|v| v.as_str()).unwrap_or("");

        if session_id.is_empty() || config_id != "model" || value.is_empty() {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32602,"message":"missing sessionId, configId, or value"})),
            };
        }

        if !self.sessions.contains_key(session_id) {
            let _ = self.restore_session_state(session_id);
        }

        let Some(session) = self.sessions.get_mut(session_id) else {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32000,"message":format!("unknown sessionId: {session_id}")})),
            };
        };

        session.model_id = Some(value.to_string());
        let conv_id = session.conversation_id.clone();
        let last_step_idx = session.last_step_idx;
        self.persist_session(session_id, conv_id.as_deref(), last_step_idx, Some(value));

        let config_options = self.config_options_json(Some(value));
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "configOptions": config_options })),
            error: None,
        }
    }

    /// Gather session state needed for prompt execution (under lock),
    /// returning extracted data so the lock can be released before subprocess spawn.
    fn prepare_prompt_state(
        &mut self,
        params: &Value,
    ) -> (String, String, Vec<String>, Option<HashSet<String>>, Option<String>, i64) {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Restore evicted session from state file if needed
        if !session_id.is_empty() && !self.sessions.contains_key(&session_id) {
            let _ = self.restore_session_state(&session_id);
        }

        let prompt_text = params
            .get("prompt")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        let clean_prompt = prompt_text.trim().to_string();

        // Take snapshot before spawning agy if we need to bind a conversation
        let snapshot = if self
            .sessions
            .get(&session_id)
            .map(|s| s.conversation_id.is_none())
            .unwrap_or(false)
        {
            Some(self.conversation_snapshot())
        } else {
            None
        };

        // Build args
        let mut args: Vec<String> = Vec::new();
        args.push("--add-dir".to_string());
        args.push(self.working_dir.clone());
        if let Ok(extra) = std::env::var("AGY_EXTRA_ARGS") {
            if let Ok(parsed) = shell_words::split(&extra) {
                args.extend(parsed);
            } else {
                eprintln!("[agy-acp] WARN: failed to parse AGY_EXTRA_ARGS, ignoring");
            }
        }
        if let Some(session) = self.sessions.get(&session_id) {
            if let Some(conv_id) = &session.conversation_id {
                args.push("--conversation".to_string());
                args.push(conv_id.clone());
            }
            if let Some(model_id) = &session.model_id {
                args.push("--model".to_string());
                args.push(model_id.clone());
            }
        }
        args.push("-p".to_string());
        args.push(clean_prompt.clone());

        let initial_conv_id = self
            .sessions
            .get(&session_id)
            .and_then(|s| s.conversation_id.clone());
        let initial_step_idx = self
            .sessions
            .get(&session_id)
            .map(|s| s.last_step_idx)
            .unwrap_or(-1);

        (session_id, clean_prompt, args, snapshot, initial_conv_id, initial_step_idx)
    }

}

/// Drop guard that sets stop_polling flag when the future is dropped (task abort safety).
struct StopGuard(Arc<AtomicBool>);
impl Drop for StopGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Output from prompt execution (used to separate lock-free execution from state update).
struct PromptOutput {
    response_lines: Vec<String>,
    /// If Some, contains (bound_conv_id, new_step_idx) for session state update.
    session_update: Option<(Option<String>, i64)>,
}

impl Adapter {
    /// Execute prompt subprocess without holding any adapter lock.
    /// This is a static method — all needed state is passed in as parameters.
    async fn execute_prompt(
        id: Value,
        session_id: &str,
        args: Vec<String>,
        snapshot: Option<HashSet<String>>,
        initial_conv_id: Option<String>,
        initial_step_idx: i64,
        working_dir: String,
        conversations_dir: PathBuf,
        cancelled: Arc<AtomicBool>,
        out_tx: mpsc::UnboundedSender<Option<String>>,
    ) -> PromptOutput {
        let spawn_result = Command::new("agy")
            .args(&args)
            .current_dir(&working_dir)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        let mut child = match spawn_result {
            Ok(child) => child,
            Err(e) => {
                return PromptOutput {
                    response_lines: vec![serde_json::to_string(&JsonRpcResponse {
                        jsonrpc: "2.0",
                        id,
                        result: None,
                        error: Some(json!({"code":-32000,"message":format!("failed to run agy: {e}")})),
                    }).unwrap()],
                    session_update: None,
                };
            }
        };

        // Drain stdout/stderr in background tasks
        let mut stdout_handle = child.stdout.take();
        let stdout_reader = tokio::spawn(async move {
            let mut buf = Vec::new();
            if let Some(mut stdout) = stdout_handle.take() {
                let _ = stdout.read_to_end(&mut buf).await;
            }
            buf
        });

        let mut stderr_handle = child.stderr.take();
        let stderr_reader = tokio::spawn(async move {
            let mut buf = Vec::new();
            if let Some(mut stderr) = stderr_handle.take() {
                let _ = stderr.read_to_end(&mut buf).await;
            }
            buf
        });

        // Set up streaming state
        let streaming_state = Arc::new(Mutex::new(StreamingState {
            conversation_id: initial_conv_id,
            base_step_idx: initial_step_idx,
            last_step_idx: initial_step_idx,
            emitted_len: HashMap::new(),
            emitted_tool_steps: HashSet::new(),
            had_updates: false,
        }));

        // Start polling thread — 100ms interval, writes through channel
        let stop_polling = Arc::new(AtomicBool::new(false));
        let poll_conversations_dir = conversations_dir.clone();
        let poll_snapshot = snapshot.clone();
        let poll_session_id = session_id.to_string();
        let poll_state = Arc::clone(&streaming_state);
        let poll_stop = Arc::clone(&stop_polling);
        let poll_tx = out_tx.clone();

        let poller = std::thread::spawn(move || {
            while !poll_stop.load(Ordering::SeqCst) {
                let lines = Self::poll_streaming_delta(
                    &poll_conversations_dir,
                    poll_snapshot.as_ref(),
                    &poll_session_id,
                    &poll_state,
                );
                for line in lines {
                    // If send fails, receiver is dropped (task cancelled) — exit
                    if poll_tx.send(Some(line)).is_err() {
                        return;
                    }
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });

        // Guard: ensure stop_polling is set if this future is dropped (task abort)
        let _stop_guard = StopGuard(Arc::clone(&stop_polling));

        // Wait for child to exit, or cancel
        let mut was_cancelled = false;
        let result = tokio::select! {
            result = child.wait() => result,
            _ = async {
                while !cancelled.load(Ordering::SeqCst) {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            } => {
                was_cancelled = true;
                let _ = child.kill().await;
                child.wait().await
            }
        };

        // Wait for stdout/stderr drains
        let _ = stdout_reader.await;
        let stderr_bytes = stderr_reader.await.unwrap_or_default();

        // Stop polling thread and do final flush
        stop_polling.store(true, Ordering::SeqCst);
        let _ = poller.join();

        // Final poll to catch any last writes
        {
            let lines = Self::poll_streaming_delta(
                &conversations_dir,
                snapshot.as_ref(),
                session_id,
                &streaming_state,
            );
            for line in lines {
                let _ = out_tx.send(Some(line));
            }
        }

        // Extract final state
        let (bound_conv_id, new_step_idx, had_updates) = {
            let guard = streaming_state.lock().unwrap();
            (
                guard.conversation_id.clone(),
                guard.last_step_idx,
                guard.had_updates,
            )
        };

        let session_update = Some((bound_conv_id.clone(), new_step_idx));

        // Build final response
        let stop_reason = if was_cancelled {
            "cancelled"
        } else if result.as_ref().map(|s| !s.success()).unwrap_or(false) {
            "error"
        } else {
            "end_turn"
        };

        match result {
            Ok(status) => {
                let stderr_text = String::from_utf8_lossy(&stderr_bytes);
                if !stderr_text.is_empty() {
                    eprintln!("[agy-acp] agy stderr: {}", stderr_text.trim_end());
                }
                if !was_cancelled && !status.success() {
                    eprintln!("[agy-acp] WARN: agy exited with status: {}", status);
                    if !had_updates {
                        let msg = if stderr_text.is_empty() {
                            format!("agy exited with status: {}", status)
                        } else {
                            format!("agy failed: {}", stderr_text.trim_end())
                        };
                        return PromptOutput {
                            response_lines: vec![serde_json::to_string(&JsonRpcResponse {
                                jsonrpc: "2.0",
                                id,
                                result: None,
                                error: Some(json!({"code":-32000,"message":msg})),
                            }).unwrap()],
                            session_update,
                        };
                    }
                }
            }
            Err(e) => {
                return PromptOutput {
                    response_lines: vec![serde_json::to_string(&JsonRpcResponse {
                        jsonrpc: "2.0",
                        id,
                        result: None,
                        error: Some(json!({"code":-32000,"message":format!("failed to wait for agy: {e}")})),
                    }).unwrap()],
                    session_update,
                };
            }
        }

        PromptOutput {
            response_lines: vec![serde_json::to_string(&JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({ "stopReason": stop_reason })),
                error: None,
            }).unwrap()],
            session_update,
        }
    }
}

#[tokio::main]
async fn main() {
    // Prefetch models in background to avoid blocking tokio worker on first session/new
    let prefetch = tokio::task::spawn_blocking(Adapter::fetch_available_models);

    let adapter = Arc::new(tokio::sync::Mutex::new(Adapter::new()));

    // Apply prefetched models
    if let Ok(models) = prefetch.await {
        let mut guard = adapter.lock().await;
        if !models.is_empty() {
            eprintln!("[agy-acp] fetched {} models from `agy models`, updating cache", models.len());
            guard.save_models_cache(&models);
            guard.available_models = Some(models);
        } else if let Some(cached) = guard.load_cached_models() {
            eprintln!("[agy-acp] `agy models` failed, using cached model list ({} models)", cached.len());
            guard.available_models = Some(cached);
        } else {
            eprintln!("[agy-acp] `agy models` failed and no cache found, using hardcoded fallback");
            guard.available_models = Some(Adapter::static_fallback_models());
        }
    }

    let active_cancellations: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>> =
        Arc::new(Mutex::new(HashMap::new()));

    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Option<String>>();

    std::thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) if !l.trim().is_empty() => {
                    if tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
                _ => {}
            }
        }
    });

    let mut stdout = io::stdout();
    let mut stdin_open = true;
    let mut pending_prompts = 0usize;

    loop {
        if !stdin_open && pending_prompts == 0 {
            break;
        }

        let line = if stdin_open {
            tokio::select! {
                output = out_rx.recv() => {
                    match output {
                        Some(Some(line)) => {
                            let _ = writeln!(stdout, "{}", line);
                            let _ = stdout.flush();
                        }
                        Some(None) => pending_prompts = pending_prompts.saturating_sub(1),
                        None => {}
                    }
                    continue;
                }
                input = rx.recv() => {
                    match input {
                        Some(line) => line,
                        None => { stdin_open = false; continue; }
                    }
                }
            }
        } else {
            match out_rx.recv().await {
                Some(Some(line)) => {
                    let _ = writeln!(stdout, "{}", line);
                    let _ = stdout.flush();
                }
                Some(None) => pending_prompts = pending_prompts.saturating_sub(1),
                None => break,
            }
            continue;
        };

        // Drain any pending output
        while let Ok(output) = out_rx.try_recv() {
            match output {
                Some(line) => {
                    let _ = writeln!(stdout, "{}", line);
                    let _ = stdout.flush();
                }
                None => pending_prompts = pending_prompts.saturating_sub(1),
            }
        }

        let req: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let id = match req.id {
            Some(id) => id,
            None => {
                // Handle notifications (no id) — only session/cancel
                if req.method.as_deref() == Some("session/cancel") {
                    let params = req.params.unwrap_or(json!({}));
                    if let Some(session_id) = params.get("sessionId").and_then(|v| v.as_str()) {
                        if let Some(cancelled) =
                            active_cancellations.lock().unwrap().get(session_id).cloned()
                        {
                            cancelled.store(true, Ordering::SeqCst);
                        }
                    }
                }
                continue;
            }
        };

        let output = match req.method.as_deref() {
            Some("initialize") => {
                let adapter = Arc::clone(&adapter);
                let out_tx = out_tx.clone();
                pending_prompts += 1;
                tokio::spawn(async move {
                    let adapter = adapter.lock().await;
                    let line =
                        serde_json::to_string(&adapter.handle_initialize(id)).unwrap();
                    let _ = out_tx.send(Some(line));
                    let _ = out_tx.send(None);
                });
                Vec::new()
            }
            Some("session/new") => {
                let adapter = Arc::clone(&adapter);
                let out_tx = out_tx.clone();
                pending_prompts += 1;
                tokio::spawn(async move {
                    let mut adapter = adapter.lock().await;
                    let line =
                        serde_json::to_string(&adapter.handle_session_new(id)).unwrap();
                    let _ = out_tx.send(Some(line));
                    let _ = out_tx.send(None);
                });
                Vec::new()
            }
            Some("session/load") => {
                let params = req.params.unwrap_or(json!({}));
                let adapter = Arc::clone(&adapter);
                let out_tx = out_tx.clone();
                pending_prompts += 1;
                tokio::spawn(async move {
                    let mut adapter = adapter.lock().await;
                    let line = serde_json::to_string(
                        &adapter.handle_session_load(id, &params),
                    )
                    .unwrap();
                    let _ = out_tx.send(Some(line));
                    let _ = out_tx.send(None);
                });
                Vec::new()
            }
            Some("session/prompt") => {
                let params = req.params.unwrap_or(json!({}));
                let session_id = params
                    .get("sessionId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let cancelled = Arc::new(AtomicBool::new(false));
                if !session_id.is_empty() {
                    active_cancellations
                        .lock()
                        .unwrap()
                        .insert(session_id.clone(), Arc::clone(&cancelled));
                }

                let adapter = Arc::clone(&adapter);
                let active_cancellations = Arc::clone(&active_cancellations);
                let out_tx = out_tx.clone();
                pending_prompts += 1;

                tokio::spawn(async move {
                    // Phase 1: acquire lock briefly to prepare state
                    let (session_id_inner, args, snapshot, initial_conv_id, initial_step_idx, working_dir, conversations_dir) = {
                        let mut adapter = adapter.lock().await;
                        let (sid, _prompt, args, snapshot, init_conv, init_idx) =
                            adapter.prepare_prompt_state(&params);
                        let wd = adapter.working_dir.clone();
                        let cd = adapter.conversations_dir.clone();
                        (sid, args, snapshot, init_conv, init_idx, wd, cd)
                    };
                    // Lock released — Phase 2: run subprocess without holding lock
                    let output = Adapter::execute_prompt(
                        id, &session_id_inner, args, snapshot, initial_conv_id, initial_step_idx,
                        working_dir, conversations_dir, cancelled, out_tx.clone(),
                    ).await;

                    // Phase 3: acquire lock briefly to update session state
                    if let Some((bound_conv_id, new_step_idx)) = output.session_update {
                        let mut adapter = adapter.lock().await;
                        if let Some(session) = adapter.sessions.get_mut(&session_id_inner) {
                            if session.conversation_id.is_none() {
                                session.conversation_id = bound_conv_id.clone();
                            }
                            if bound_conv_id.is_some() {
                                session.last_step_idx = new_step_idx;
                            }
                        }
                        if bound_conv_id.is_some() {
                            let model_id = adapter.sessions.get(&session_id_inner).and_then(|s| s.model_id.clone());
                            adapter.persist_session(&session_id_inner, bound_conv_id.as_deref(), new_step_idx, model_id.as_deref());
                        }
                    }

                    if !session_id.is_empty() {
                        active_cancellations.lock().unwrap().remove(&session_id);
                    }
                    for line in output.response_lines {
                        let _ = out_tx.send(Some(line));
                    }
                    let _ = out_tx.send(None);
                });
                Vec::new()
            }
            Some("session/setConfigOption") | Some("session/set_config_option") => {
                let params = req.params.unwrap_or(json!({}));
                let adapter = Arc::clone(&adapter);
                let out_tx = out_tx.clone();
                pending_prompts += 1;
                tokio::spawn(async move {
                    let mut adapter = adapter.lock().await;
                    let line = serde_json::to_string(
                        &adapter.handle_session_set_config_option(id, &params),
                    )
                    .unwrap();
                    let _ = out_tx.send(Some(line));
                    let _ = out_tx.send(None);
                });
                Vec::new()
            }
            Some("session/cancel") => {
                let params = req.params.unwrap_or(json!({}));
                if let Some(session_id) = params.get("sessionId").and_then(|v| v.as_str()) {
                    if let Some(cancelled) =
                        active_cancellations.lock().unwrap().get(session_id).cloned()
                    {
                        cancelled.store(true, Ordering::SeqCst);
                    }
                }
                vec![serde_json::to_string(&JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({})),
                    error: None,
                })
                .unwrap()]
            }
            Some(method) => {
                vec![serde_json::to_string(&JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: None,
                    error: Some(
                        json!({"code":-32601,"message":format!("method not found: {method}")}),
                    ),
                })
                .unwrap()]
            }
            None => continue,
        };

        for line in output {
            let _ = writeln!(stdout, "{}", line);
        }
        let _ = stdout.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_from_step_payload_field20_field1() {
        let mut inner = Vec::new();
        inner.push(0x0A); inner.push(0x05);
        inner.extend_from_slice(b"hello");

        let mut blob = Vec::new();
        blob.push(0x08); blob.push(0x0F);
        blob.push(0xA2); blob.push(0x01);
        blob.push(inner.len() as u8);
        blob.extend_from_slice(&inner);
        assert_eq!(Adapter::extract_text_from_step_payload(&blob), Some("hello".to_string()));
    }

    #[test]
    fn test_extract_text_returns_none_without_field20() {
        let blob = vec![0x08, 0x03];
        assert_eq!(Adapter::extract_text_from_step_payload(&blob), None);
    }

    #[test]
    fn test_extract_text_multiline() {
        let text = b"Safe memory rules\nCompiler points out the flaws\nFast and fearless code";
        let mut inner = Vec::new();
        inner.push(0x0A);
        inner.push(text.len() as u8);
        inner.extend_from_slice(text);

        let mut blob = Vec::new();
        blob.push(0x08); blob.push(0x01);
        blob.push(0xA2); blob.push(0x01);
        blob.push(inner.len() as u8);
        blob.extend_from_slice(&inner);
        assert_eq!(
            Adapter::extract_text_from_step_payload(&blob),
            Some("Safe memory rules\nCompiler points out the flaws\nFast and fearless code".to_string())
        );
    }

    #[test]
    fn test_read_varint() {
        assert_eq!(Adapter::read_varint(&[0x05]), Some((5, 1)));
        assert_eq!(Adapter::read_varint(&[0xAC, 0x02]), Some((300, 2)));
        assert_eq!(Adapter::read_varint(&[]), None);
    }

    #[test]
    fn test_initialize_advertises_load_session_support() {
        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: "/tmp".to_string(),
            conversations_dir: PathBuf::from("/tmp"),
            state_file: PathBuf::from("/tmp/sessions.json"),
            available_models: Some(vec![]),
        };
        let response = adapter.handle_initialize(json!(1));
        assert_eq!(
            response
                .result
                .as_ref()
                .and_then(|r| r.get("agentCapabilities"))
                .and_then(|c| c.get("loadSession"))
                .and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_session_load_restores_persisted_session() {
        let root = std::env::temp_dir().join(format!("agy-acp-load-{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&root);

        let mut adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
            available_models: Some(vec![]),
        };
        adapter.persist_session("sess-1", Some("conv-abc"), 5, None);

        let response = adapter.handle_session_load(json!(7), &json!({"sessionId": "sess-1"}));
        assert!(response.error.is_none());
        assert_eq!(
            adapter.sessions.get("sess-1").and_then(|s| s.conversation_id.as_deref()),
            Some("conv-abc")
        );
        assert_eq!(
            adapter.sessions.get("sess-1").map(|s| s.last_step_idx),
            Some(5)
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_session_load_rejects_unknown_session() {
        let root = std::env::temp_dir().join(format!("agy-acp-missing-{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&root);

        let mut adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
            available_models: Some(vec![]),
        };

        let response = adapter.handle_session_load(json!(9), &json!({"sessionId": "missing"}));
        assert!(response.result.is_none());
        assert_eq!(
            response.error.as_ref().and_then(|e| e.get("message")).and_then(|m| m.as_str()),
            Some("unknown sessionId: missing")
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_snapshot_detects_db_conversations() {
        let root = std::env::temp_dir().join(format!("agy-acp-db-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();
        fs::write(conv_dir.join("existing.db"), b"old").unwrap();

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir.clone(),
            state_file: root.join("sessions.json"),
            available_models: Some(vec![]),
        };

        let before = adapter.conversation_snapshot();
        assert!(before.contains("existing"));

        fs::write(conv_dir.join("new-conv.db"), b"new").unwrap();
        fs::write(conv_dir.join("new-conv.db-wal"), b"wal").unwrap();
        fs::write(conv_dir.join("new-conv.db-shm"), b"shm").unwrap();

        assert_eq!(
            adapter.new_conversation_id(&before),
            Some("new-conv".to_string())
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_snapshot_ignores_multiple_new_files() {
        let root = std::env::temp_dir().join(format!("agy-acp-multi-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir.clone(),
            state_file: root.join("sessions.json"),
            available_models: Some(vec![]),
        };

        let before = adapter.conversation_snapshot();
        fs::write(conv_dir.join("a.db"), b"").unwrap();
        fs::write(conv_dir.join("b.db"), b"").unwrap();

        assert_eq!(adapter.new_conversation_id(&before), None);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_persist_and_restore_session() {
        let root = std::env::temp_dir().join(format!("agy-acp-state-{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&root);

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
            available_models: Some(vec![]),
        };

        adapter.persist_session("sess-1", Some("conv-abc"), 7, None);
        let restored = adapter.restore_session("sess-1");
        assert_eq!(restored, Some(("conv-abc".to_string(), 7, None)));

        let missing = adapter.restore_session("sess-unknown");
        assert_eq!(missing, None);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O — requires real SQLite DB
    fn test_read_response_from_db() {
        let root = std::env::temp_dir().join(format!("agy-acp-sqlite-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();

        let db_path = conv_dir.join("test-conv.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE steps (
                idx INTEGER PRIMARY KEY,
                step_type INTEGER NOT NULL DEFAULT 0,
                status INTEGER NOT NULL DEFAULT 0,
                has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
                metadata BLOB,
                error_details BLOB,
                permissions BLOB,
                task_details BLOB,
                render_info BLOB,
                step_payload BLOB,
                step_format INTEGER NOT NULL DEFAULT 0
            )"
        ).unwrap();

        let mut inner = Vec::new();
        inner.push(0x0A); inner.push(11);
        inner.extend_from_slice(b"hello world");
        let mut payload = Vec::new();
        payload.push(0x08); payload.push(0x0F);
        payload.push(0xA2); payload.push(0x01);
        payload.push(inner.len() as u8);
        payload.extend_from_slice(&inner);

        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![1i64, payload],
        ).unwrap();

        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 14, ?2)",
            rusqlite::params![2i64, vec![0x08u8, 0x0E]],
        ).unwrap();
        drop(conn);

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir,
            state_file: root.join("sessions.json"),
            available_models: Some(vec![]),
        };

        let result = adapter.read_response_from_db("test-conv", -1);
        assert_eq!(result, Some(("hello world".to_string(), 1)));

        let result = adapter.read_response_from_db("test-conv", 1);
        assert_eq!(result, None);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_streaming_poll_emits_delta() {
        let root = std::env::temp_dir().join(format!("agy-acp-stream-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();

        let db_path = conv_dir.join("stream-conv.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE steps (
                idx INTEGER PRIMARY KEY,
                step_type INTEGER NOT NULL DEFAULT 0,
                status INTEGER NOT NULL DEFAULT 0,
                has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
                metadata BLOB, error_details BLOB, permissions BLOB,
                task_details BLOB, render_info BLOB, step_payload BLOB,
                step_format INTEGER NOT NULL DEFAULT 0
            )"
        ).unwrap();

        fn make_payload(text: &str) -> Vec<u8> {
            let text_bytes = text.as_bytes();
            let mut inner = vec![0x0A];
            let mut len = text_bytes.len();
            loop {
                if len < 128 { inner.push(len as u8); break; }
                inner.push((len as u8 & 0x7F) | 0x80);
                len >>= 7;
            }
            inner.extend_from_slice(text_bytes);
            let mut outer = vec![0xA2, 0x01];
            let mut ilen = inner.len();
            loop {
                if ilen < 128 { outer.push(ilen as u8); break; }
                outer.push((ilen as u8 & 0x7F) | 0x80);
                ilen >>= 7;
            }
            outer.extend(inner);
            outer
        }

        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![1i64, make_payload("hello")],
        ).unwrap();

        let state = Arc::new(Mutex::new(StreamingState {
            conversation_id: Some("stream-conv".to_string()),
            base_step_idx: -1,
            last_step_idx: -1,
            emitted_len: HashMap::new(),
            emitted_tool_steps: HashSet::new(),
            had_updates: false,
        }));

        // First poll — should get "hello"
        let lines = Adapter::poll_streaming_delta(&conv_dir, None, "sess-1", &state);
        assert_eq!(lines.len(), 1);
        let msg: Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(msg["params"]["update"]["content"]["text"], "hello");

        // Second poll — no new data, no output
        let lines = Adapter::poll_streaming_delta(&conv_dir, None, "sess-1", &state);
        assert!(lines.is_empty());

        // Simulate text growing (agy appending to same step)
        conn.execute(
            "UPDATE steps SET step_payload = ?1 WHERE idx = 1",
            rusqlite::params![make_payload("hello world")],
        ).unwrap();

        // Third poll — should get " world" (delta)
        let lines = Adapter::poll_streaming_delta(&conv_dir, None, "sess-1", &state);
        assert_eq!(lines.len(), 1);
        let msg: Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(msg["params"]["update"]["content"]["text"], " world");

        // Verify state
        let guard = state.lock().unwrap();
        assert_eq!(guard.last_step_idx, 1);
        assert!(guard.had_updates);

        drop(conn);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_read_response_multi_step_no_skip_no_duplicate() {
        let root = std::env::temp_dir().join(format!("agy-acp-multi-step-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();

        let db_path = conv_dir.join("multi.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE steps (
                idx INTEGER PRIMARY KEY,
                step_type INTEGER NOT NULL DEFAULT 0,
                status INTEGER NOT NULL DEFAULT 0,
                has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
                metadata BLOB, error_details BLOB, permissions BLOB,
                task_details BLOB, render_info BLOB, step_payload BLOB,
                step_format INTEGER NOT NULL DEFAULT 0
            )"
        ).unwrap();

        fn make_payload(text: &str) -> Vec<u8> {
            let text_bytes = text.as_bytes();
            let mut inner = vec![0x0A];
            let mut len = text_bytes.len();
            loop {
                if len < 128 { inner.push(len as u8); break; }
                inner.push((len as u8 & 0x7F) | 0x80);
                len >>= 7;
            }
            inner.extend_from_slice(text_bytes);
            let mut outer = vec![0xA2, 0x01];
            let mut ilen = inner.len();
            loop {
                if ilen < 128 { outer.push(ilen as u8); break; }
                outer.push((ilen as u8 & 0x7F) | 0x80);
                ilen >>= 7;
            }
            outer.extend(inner);
            outer
        }

        conn.execute("INSERT INTO steps (idx, step_type, step_payload) VALUES (1, 0, X'0801')", []).unwrap();
        conn.execute("INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![2i64, make_payload("hello")]).unwrap();
        conn.execute("INSERT INTO steps (idx, step_type, step_payload) VALUES (3, 0, X'0802')", []).unwrap();
        conn.execute("INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![4i64, make_payload("world")]).unwrap();
        conn.execute("INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![5i64, make_payload("line1\nline2\nline3")]).unwrap();
        drop(conn);

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir,
            state_file: root.join("sessions.json"),
            available_models: Some(vec![]),
        };

        let result = adapter.read_response_from_db("multi", -1);
        assert_eq!(result, Some(("hello\nworld\nline1\nline2\nline3".to_string(), 5)));

        let result = adapter.read_response_from_db("multi", 2);
        assert_eq!(result, Some(("world\nline1\nline2\nline3".to_string(), 5)));

        let result = adapter.read_response_from_db("multi", 4);
        assert_eq!(result, Some(("line1\nline2\nline3".to_string(), 5)));

        let result = adapter.read_response_from_db("multi", 5);
        assert_eq!(result, None);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_read_response_missing_steps_table() {
        let root = std::env::temp_dir().join(format!("agy-acp-noschema-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();

        let db_path = conv_dir.join("empty.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch("CREATE TABLE other (id INTEGER)").unwrap();
        drop(conn);

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir,
            state_file: root.join("sessions.json"),
            available_models: Some(vec![]),
        };

        let result = adapter.read_response_from_db("empty", -1);
        assert_eq!(result, None);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn test_is_narration_true() {
        assert!(Adapter::is_narration("I will fetch the latest commits."));
        assert!(Adapter::is_narration("I will fetch the latest commits.\nI will check the diff."));
        assert!(Adapter::is_narration("I will read the file.\n\nI will analyze the output."));
    }

    #[test]
    fn test_is_narration_false() {
        assert!(!Adapter::is_narration("Here is the result."));
        assert!(!Adapter::is_narration("I will fetch the commits.\nHere is the result."));
        assert!(!Adapter::is_narration(""));
    }

    #[test]
    fn test_filter_narration_drops_leading_narration() {
        // Narration is now skipped by default
        std::env::remove_var("OPENAB_SHOW_NARRATION");
        let parts = vec![
            "I will fetch the latest commits.\nI will check the diff.".to_string(),
            "I will read the file.".to_string(),
            "The fix is confirmed! LGTM ✅".to_string(),
        ];
        let result = Adapter::filter_narration(&parts);
        assert_eq!(result, "The fix is confirmed! LGTM ✅");
    }

    #[test]
    fn test_filter_narration_preserves_content_after_first_non_narration() {
        std::env::remove_var("OPENAB_SHOW_NARRATION");
        let parts = vec![
            "I will check things.".to_string(),
            "Here is my analysis.".to_string(),
            "I will also note this is fine.".to_string(),
        ];
        let result = Adapter::filter_narration(&parts);
        assert_eq!(result, "Here is my analysis.\nI will also note this is fine.");
    }

    #[test]
    fn test_filter_narration_show_mode() {
        std::env::set_var("OPENAB_SHOW_NARRATION", "1");
        let parts = vec![
            "I will fetch commits.".to_string(),
            "Final answer here.".to_string(),
        ];
        let result = Adapter::filter_narration(&parts);
        assert_eq!(result, "I will fetch commits.\nFinal answer here.");
        std::env::remove_var("OPENAB_SHOW_NARRATION");
    }

    #[test]
    fn test_filter_narration_default_skips() {
        std::env::remove_var("OPENAB_SHOW_NARRATION");
        let parts = vec![
            "I will fetch commits.".to_string(),
            "Final answer here.".to_string(),
        ];
        let result = Adapter::filter_narration(&parts);
        assert_eq!(result, "Final answer here.");
    }

    #[test]
    fn test_filter_narration_single_part_unchanged() {
        let parts = vec!["I will do something.".to_string()];
        let result = Adapter::filter_narration(&parts);
        // Single part is never filtered (would leave nothing)
        assert_eq!(result, "I will do something.");
    }

    #[test]
    fn test_filter_narration_all_narration_keeps_last() {
        std::env::remove_var("OPENAB_SHOW_NARRATION");
        let parts = vec![
            "I will fetch the file.".to_string(),
            "I will check the output.".to_string(),
            "I will verify the fix.".to_string(),
        ];
        let result = Adapter::filter_narration(&parts);
        assert_eq!(result, "I will verify the fix.");
    }

    #[test]
    fn test_json_rpc_id_as_string() {
        // Verify that string IDs are handled correctly
        let json_str = r#"{"jsonrpc":"2.0","id":"abc-123","method":"initialize","params":{}}"#;
        let req: JsonRpcRequest = serde_json::from_str(json_str).unwrap();
        assert_eq!(req.id, Some(json!("abc-123")));
    }

    #[test]
    fn test_json_rpc_id_as_number() {
        let json_str = r#"{"jsonrpc":"2.0","id":42,"method":"initialize","params":{}}"#;
        let req: JsonRpcRequest = serde_json::from_str(json_str).unwrap();
        assert_eq!(req.id, Some(json!(42)));
    }

    /// Check auth is available: either GEMINI_API_KEY env var or local keyring.
    fn prepare_auth() -> bool {
        if std::env::var("GEMINI_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            eprintln!("[e2e] Using GEMINI_API_KEY");
            return true;
        }
        let home = std::env::var("HOME").unwrap_or_default();
        let settings = format!("{}/.gemini/antigravity-cli/settings.json", home);
        if std::path::Path::new(&settings).exists() {
            eprintln!("[e2e] Using local auth (keyring)");
            return true;
        }
        eprintln!("SKIP: No GEMINI_API_KEY and no local auth found");
        false
    }

    /// E2E test: spawns agy-acp, sends initialize → session/new → session/prompt,
    /// and verifies the response contains expected text from real agy.
    #[test]
    #[ignore]
    fn test_e2e_agy_acp_full_round_trip() {
        use std::io::{BufRead, BufReader, Write};
        use std::process::{Command, Stdio};
        use std::time::Duration;

        if !prepare_auth() { return; }

        let agy_check = Command::new("agy").arg("--help").output();
        if agy_check.is_err() || !agy_check.unwrap().status.success() {
            eprintln!("SKIP: agy not found in PATH");
            return;
        }

        let binary = std::env::current_dir().unwrap().join("target/release/agy-acp");
        if !binary.exists() { panic!("Run `cargo build --release` first"); }

        let mut child = Command::new(&binary)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn().expect("failed to spawn agy-acp");

        let mut stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let mut reader = BufReader::new(stdout);

        let mut send_and_recv = |msg: &str| -> String {
            writeln!(stdin, "{}", msg).unwrap();
            stdin.flush().unwrap();
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            line
        };

        let resp = send_and_recv(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#);
        let init: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(init["result"]["protocolVersion"], 1);

        let resp = send_and_recv(r#"{"jsonrpc":"2.0","id":2,"method":"session/new","params":{}}"#);
        let session: Value = serde_json::from_str(&resp).unwrap();
        let session_id = session["result"]["sessionId"].as_str().unwrap();
        assert!(!session_id.is_empty());

        let prompt_msg = format!(
            r#"{{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{{"sessionId":"{}","prompt":[{{"type":"text","text":"Reply with exactly one word: PONG"}}]}}}}"#,
            session_id
        );
        writeln!(stdin, "{}", prompt_msg).unwrap();
        stdin.flush().unwrap();

        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        let mut got_notification = false;
        let mut response_text = String::new();
        loop {
            if std::time::Instant::now() > deadline { panic!("Timed out"); }
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line.is_empty() { std::thread::sleep(Duration::from_millis(100)); continue; }
            let msg: Value = serde_json::from_str(line.trim()).unwrap();
            if msg.get("method") == Some(&json!("session/update")) {
                got_notification = true;
                if let Some(text) = msg["params"]["update"]["content"]["text"].as_str() {
                    response_text.push_str(text);
                }
            }
            if msg.get("id") == Some(&json!(3)) {
                assert!(msg["error"].is_null(), "Got error: {}", msg["error"]);
                break;
            }
        }

        drop(stdin);
        let _ = child.wait();

        assert!(got_notification, "Expected session/update notification");
        let lower = response_text.to_lowercase();
        assert!(lower.contains("pong"), "Expected 'PONG' in response, got: '{}'", response_text);
    }

    /// Helper: spawn agy-acp, return (stdin, reader, child)
    fn spawn_agy_acp() -> Option<(std::process::ChildStdin, std::io::BufReader<std::process::ChildStdout>, std::process::Child)> {
        use std::io::BufReader;
        use std::process::{Command, Stdio};

        if !prepare_auth() { return None; }
        let agy_check = Command::new("agy").arg("--help").output();
        if agy_check.is_err() || !agy_check.unwrap().status.success() {
            eprintln!("SKIP: agy not found in PATH");
            return None;
        }
        let binary = std::env::current_dir().unwrap().join("target/release/agy-acp");
        if !binary.exists() { panic!("Run `cargo build --release` first"); }

        let mut child = Command::new(&binary)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn().expect("failed to spawn agy-acp");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        Some((stdin, BufReader::new(stdout), child))
    }

    /// Helper: send JSON-RPC and read one response line
    fn send_recv(stdin: &mut std::process::ChildStdin, reader: &mut std::io::BufReader<std::process::ChildStdout>, msg: &str) -> String {
        use std::io::{BufRead, Write};
        writeln!(stdin, "{}", msg).unwrap();
        stdin.flush().unwrap();
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        line
    }

    /// Helper: send a prompt and wait for the response (notifications + final reply)
    fn send_prompt_wait(stdin: &mut std::process::ChildStdin, reader: &mut std::io::BufReader<std::process::ChildStdout>, id: u64, session_id: &str, text: &str) -> (Option<String>, Value) {
        use std::io::{BufRead, Write};
        use std::time::Duration;

        let msg = format!(
            r#"{{"jsonrpc":"2.0","id":{},"method":"session/prompt","params":{{"sessionId":"{}","prompt":[{{"type":"text","text":"{}"}}]}}}}"#,
            id, session_id, text
        );
        writeln!(stdin, "{}", msg).unwrap();
        stdin.flush().unwrap();

        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        let mut collected_text = String::new();
        loop {
            if std::time::Instant::now() > deadline { panic!("Timed out"); }
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line.is_empty() { std::thread::sleep(Duration::from_millis(100)); continue; }
            let msg: Value = serde_json::from_str(line.trim()).unwrap();
            if msg.get("method") == Some(&json!("session/update")) {
                if let Some(text) = msg["params"]["update"]["content"]["text"].as_str() {
                    collected_text.push_str(text);
                }
            }
            if msg.get("id") == Some(&json!(id)) {
                let text = if collected_text.is_empty() { None } else { Some(collected_text) };
                return (text, msg);
            }
        }
    }

    #[test]
    #[ignore]
    fn test_e2e_multi_turn() {
        let Some((mut stdin, mut reader, mut child)) = spawn_agy_acp() else { return };

        send_recv(&mut stdin, &mut reader, r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#);
        let resp = send_recv(&mut stdin, &mut reader, r#"{"jsonrpc":"2.0","id":2,"method":"session/new","params":{}}"#);
        let session_id = serde_json::from_str::<Value>(&resp).unwrap()["result"]["sessionId"].as_str().unwrap().to_string();

        let (text1, resp1) = send_prompt_wait(&mut stdin, &mut reader, 3, &session_id, "Remember this word: BANANA. Reply OK.");
        assert!(resp1["error"].is_null(), "Turn 1 error: {}", resp1["error"]);
        assert!(text1.is_some());

        let (text2, resp2) = send_prompt_wait(&mut stdin, &mut reader, 4, &session_id, "What word did I ask you to remember? Reply with just that word.");
        assert!(resp2["error"].is_null(), "Turn 2 error: {}", resp2["error"]);
        let reply = text2.unwrap_or_default().to_lowercase();
        assert!(reply.contains("banana"), "Expected 'BANANA' in multi-turn reply, got: '{}'", reply);

        drop(stdin);
        let _ = child.wait();
    }

    #[test]
    #[ignore]
    fn test_e2e_session_load() {
        let Some((mut stdin, mut reader, mut child)) = spawn_agy_acp() else { return };

        send_recv(&mut stdin, &mut reader, r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#);
        let resp = send_recv(&mut stdin, &mut reader, r#"{"jsonrpc":"2.0","id":2,"method":"session/new","params":{}}"#);
        let session_id = serde_json::from_str::<Value>(&resp).unwrap()["result"]["sessionId"].as_str().unwrap().to_string();

        let (_text, resp1) = send_prompt_wait(&mut stdin, &mut reader, 3, &session_id, "Reply with exactly: FIRST_TURN");
        assert!(resp1["error"].is_null(), "First turn error: {}", resp1["error"]);

        let (text2, resp2) = send_prompt_wait(&mut stdin, &mut reader, 4, &session_id, "Reply with exactly one word: SECOND");
        assert!(resp2["error"].is_null(), "Second turn error: {}", resp2["error"]);
        assert!(text2.is_some(), "Expected response on continued session");

        drop(stdin);
        let _ = child.wait();
    }

    #[test]
    #[ignore]
    fn test_e2e_error_paths() {
        let Some((mut stdin, mut reader, mut child)) = spawn_agy_acp() else { return };

        send_recv(&mut stdin, &mut reader, r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#);

        let resp = send_recv(&mut stdin, &mut reader, r#"{"jsonrpc":"2.0","id":2,"method":"session/load","params":{"sessionId":"non-existent-session"}}"#);
        let val: Value = serde_json::from_str(&resp).unwrap();
        assert!(!val["error"].is_null(), "Expected error for unknown session");

        let resp = send_recv(&mut stdin, &mut reader, r#"{"jsonrpc":"2.0","id":3,"method":"bogus/method","params":{}}"#);
        let val: Value = serde_json::from_str(&resp).unwrap();
        assert!(!val["error"].is_null(), "Expected error for unknown method");

        drop(stdin);
        let _ = child.wait();
    }
}
