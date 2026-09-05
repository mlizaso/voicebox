// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod accessibility;
mod audio_capture;
mod audio_output;
mod clipboard;
mod focus_capture;
#[cfg(desktop)]
mod hotkey_monitor;
mod input_monitoring;
#[cfg(desktop)]
mod key_codes;
mod keyboard_layout;
mod speak_monitor;
mod synthetic_keys;

use std::sync::Mutex;
use tauri::{
    command, Emitter, Listener, Manager, PhysicalPosition, RunEvent, State, WebviewUrl,
    WebviewWindowBuilder, WindowEvent,
};
use tauri_plugin_shell::ShellExt;
use tokio::sync::mpsc;

pub const DICTATE_WINDOW_LABEL: &str = "dictate";
const DICTATE_WINDOW_WIDTH: f64 = 420.0;
const DICTATE_WINDOW_HEIGHT: f64 = 64.0;

/// Create the floating dictate webview hidden. The HotkeyMonitor shows it on
/// chord-start; the frontend hides it when the capture pipeline finishes.
/// Building it at setup avoids a race where the first chord or agent-speech
/// event fires before the webview subscribes to the `dictate:*` events.
#[cfg(desktop)]
fn build_dictate_window(app: &tauri::AppHandle) -> tauri::Result<tauri::WebviewWindow> {
    let window = WebviewWindowBuilder::new(
        app,
        DICTATE_WINDOW_LABEL,
        WebviewUrl::App("?view=dictate".into()),
    )
    .title("Voicebox Dictate")
    .inner_size(DICTATE_WINDOW_WIDTH, DICTATE_WINDOW_HEIGHT)
    .decorations(false)
    .transparent(true)
    .always_on_top(true)
    // Follow the user across macOS Spaces / virtual desktops instead of
    // being pinned to the Space where the window was first created.
    .visible_on_all_workspaces(true)
    .skip_taskbar(true)
    .resizable(false)
    .shadow(false)
    .visible(false)
    .build()?;

    if let Some(monitor) = window.current_monitor()? {
        let monitor_size = monitor.size();
        let win_size = window.outer_size()?;
        let x = (monitor_size.width as i32 - win_size.width as i32) / 2;
        let y = (monitor_size.height as f64 * 0.04) as i32;
        window.set_position(PhysicalPosition::new(x, y))?;
    }

    Ok(window)
}

/// Position, undo click-through, and show the dictate pill window.
///
/// The hide path parks the window at (-10_000, -10_000) and toggles
/// `ignore_cursor_events(true)` so invisible click targets don't leak; we
/// undo both here. Mirrors the logic the hotkey_monitor's
/// `Effect::StartRecording` path runs, minus the focus snapshot — this is
/// for agent-initiated speech, not dictation, so there's no focused text
/// field to paste into.
/// Build the pill webview if it doesn't exist yet. Idempotent — used by
/// agent-speech to prime the webview on speak-start so its listeners can
/// register before the actual show arrives from `audio.onplaying`.
#[cfg(desktop)]
pub fn ensure_dictate_window(app: &tauri::AppHandle) {
    if app.get_webview_window(DICTATE_WINDOW_LABEL).is_none() {
        if let Err(e) = build_dictate_window(app) {
            eprintln!("ensure_dictate_window: failed to build pill: {e}");
        }
    }
}

#[cfg(desktop)]
pub fn show_dictate_window(app: &tauri::AppHandle) {
    // Build on demand so agent-initiated speech works before the user has
    // enabled the global hotkey (the hotkey path is the other place this
    // window gets built, see `enable_hotkey`).
    let window = match app.get_webview_window(DICTATE_WINDOW_LABEL) {
        Some(w) => w,
        None => match build_dictate_window(app) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("show_dictate_window: failed to build pill window: {e}");
                return;
            }
        },
    };
    // current_monitor() returns None when the window has been parked
    // off any display by the hide path; fall back to the primary.
    let monitor = window
        .current_monitor()
        .ok()
        .flatten()
        .or_else(|| window.primary_monitor().ok().flatten());
    if let Some(monitor) = monitor {
        let monitor_pos = monitor.position();
        let monitor_size = monitor.size();
        if let Ok(win_size) = window.outer_size() {
            let x = monitor_pos.x + (monitor_size.width as i32 - win_size.width as i32) / 2;
            let y = monitor_pos.y + (monitor_size.height as f64 * 0.04) as i32;
            let _ = window.set_position(PhysicalPosition::new(x, y));
        }
    }
    // Skip on Linux: tao's CursorIgnoreEvents handler unwraps the GdkWindow,
    // which is None until the window is first shown, aborting the process.
    // The click-through toggle is a macOS workaround and is never set on Linux.
    #[cfg(not(target_os = "linux"))]
    let _ = window.set_ignore_cursor_events(false);
    let _ = window.show();
}

pub const SERVER_PORT: u16 = 17493;

#[derive(Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ClientConnection {
    connection_id: String,
    server_url: String,
    remote_api_token: String,
    mode: String,
}

struct ServerState {
    // The main webview owns this snapshot. Never write it to disk or logs.
    client_connection: Mutex<Option<ClientConnection>>,
    child: Mutex<Option<tauri_plugin_shell::process::CommandChild>>,
    start_lock: tokio::sync::Mutex<()>,
    speak_monitor: Mutex<Option<tauri::async_runtime::JoinHandle<()>>>,
    server_pid: Mutex<Option<u32>>,
    keep_running_on_close: Mutex<bool>,
    models_dir: Mutex<Option<String>>,
    remote_mode: Mutex<bool>,
    remote_api_token: Mutex<Option<String>>,
    /// Override the backend selection: Some("cpu") forces the CPU sidecar even
    /// when GPU binaries exist (solving the Windows catch-22 where an active
    /// .exe cannot be deleted), while Some("cuda")/Some("rocm") pin a specific
    /// GPU variant when more than one is installed. None uses the on-disk
    /// default (ROCm preferred, then CUDA). Persisted to disk so the choice
    /// survives an app restart.
    backend_override: Mutex<Option<String>>,
}

#[command]
fn set_client_connection(
    window: tauri::WebviewWindow,
    state: State<ServerState>,
    connection: ClientConnection,
) -> Result<(), String> {
    if window.label() != "main" {
        return Err("Only the main window can change the connection".into());
    }
    *state.client_connection.lock().map_err(|e| e.to_string())? = Some(connection);
    Ok(())
}

#[command]
fn get_client_connection(state: State<ServerState>) -> Result<Option<ClientConnection>, String> {
    Ok(state
        .client_connection
        .lock()
        .map_err(|e| e.to_string())?
        .clone())
}

fn validate_remote_api_token(token: &str) -> Result<(), String> {
    if token.len() < 32 || token.len() > 512 {
        return Err("Remote API token must be between 32 and 512 characters".to_string());
    }
    if !token
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~'))
    {
        return Err("Remote API token must use URL-safe ASCII characters".to_string());
    }
    Ok(())
}

fn insecure_remote_http_allowed() -> Result<bool, String> {
    match std::env::var("VOICEBOX_ALLOW_INSECURE_REMOTE_HTTP") {
        Ok(value) if value.is_empty() || value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Ok(_) => Err("VOICEBOX_ALLOW_INSECURE_REMOTE_HTTP must be either 0 or 1".to_string()),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("VOICEBOX_ALLOW_INSECURE_REMOTE_HTTP must contain valid Unicode".to_string())
        }
    }
}

fn backend_override_file(data_dir: &std::path::Path) -> std::path::PathBuf {
    data_dir.join("backend_override")
}

fn read_persisted_backend_override(data_dir: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(backend_override_file(data_dir))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn write_persisted_backend_override(data_dir: &std::path::Path, value: Option<&str>) {
    let path = backend_override_file(data_dir);
    match value {
        Some(v) => {
            let _ = std::fs::create_dir_all(data_dir);
            if let Err(e) = std::fs::write(&path, v) {
                println!("Failed to persist backend override: {}", e);
            }
        }
        None => {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Run `<exe> --version` with a 10-second timeout to avoid hanging Tauri startup.
/// Returns the last whitespace-delimited token from stdout (e.g. "0.4.4"), or None on any failure.
async fn probe_binary_version(exe: &std::path::Path, cwd: &std::path::Path) -> Option<String> {
    let mut cmd = tokio::process::Command::new(exe);
    cmd.arg("--version").current_dir(cwd).kill_on_drop(true);

    match tokio::time::timeout(std::time::Duration::from_secs(10), cmd.output()).await {
        Ok(Ok(output)) => {
            let s = String::from_utf8_lossy(&output.stdout);
            s.trim().split_whitespace().last().map(String::from)
        }
        Ok(Err(e)) => {
            println!("Version probe failed: {}", e);
            None
        }
        Err(_) => {
            println!("Version probe timed out after 10s");
            None
        }
    }
}

fn server_ready(app: &tauri::AppHandle, state: &ServerState) -> String {
    let mut monitor = state.speak_monitor.lock().unwrap();
    if monitor.is_none() {
        *monitor = Some(speak_monitor::spawn_speak_monitor(app.clone()));
    }
    format!("http://127.0.0.1:{SERVER_PORT}")
}

fn clear_managed_server(app: &tauri::AppHandle, process_pid: u32, terminate: bool) {
    let state = app.state::<ServerState>();
    let mut pid = state.server_pid.lock().unwrap();
    // An old output reader must never clear a replacement child.
    if *pid != Some(process_pid) {
        return;
    }
    *pid = None;
    let child = state.child.lock().unwrap().take();
    if terminate {
        if let Some(child) = child {
            let _ = child.kill();
        }
    }
    if let Some(monitor) = state.speak_monitor.lock().unwrap().take() {
        monitor.abort();
    }
    let _ = app.emit("server-stopped", ());
}

#[command]
async fn start_server(
    app: tauri::AppHandle,
    state: State<'_, ServerState>,
    remote: Option<bool>,
    models_dir: Option<String>,
    remote_api_token: Option<String>,
    allow_external: Option<bool>,
) -> Result<String, String> {
    let _start = state.start_lock.lock().await;
    // Store models_dir for use on restart (empty string means reset to default)
    if let Some(ref dir) = models_dir {
        if dir.is_empty() {
            *state.models_dir.lock().unwrap() = None;
        } else {
            *state.models_dir.lock().unwrap() = Some(dir.clone());
        }
    }
    if let Some(remote_mode) = remote {
        *state.remote_mode.lock().unwrap() = remote_mode;
    }
    if let Some(token) = remote_api_token {
        if token.is_empty() {
            *state.remote_api_token.lock().unwrap() = None;
        } else {
            validate_remote_api_token(&token)?;
            *state.remote_api_token.lock().unwrap() = Some(token);
        }
    }
    let is_remote = *state.remote_mode.lock().unwrap();
    let effective_remote_api_token = state
        .remote_api_token
        .lock()
        .unwrap()
        .clone()
        .or_else(|| std::env::var("VOICEBOX_REMOTE_API_TOKEN").ok());
    if let Some(ref token) = effective_remote_api_token {
        validate_remote_api_token(token)?;
    }
    let bind_remote = is_remote && insecure_remote_http_allowed()?;
    if is_remote && !bind_remote {
        eprintln!(
            "Network access requested, but direct HTTP binding is disabled; keeping the server on loopback"
        );
    }
    if bind_remote && effective_remote_api_token.is_none() {
        return Err(
            "Remote network access requires a 32+ character Remote API token in Settings"
                .to_string(),
        );
    }
    // Check if server is already running (managed by this app instance)
    if state.child.lock().unwrap().is_some() {
        return Ok(server_ready(&app, &state));
    }

    // A process name or a public health response cannot authenticate a listener.
    // Reuse requires the user's explicit "Connect to my running server" action.
    if std::net::TcpStream::connect_timeout(
        &format!("127.0.0.1:{SERVER_PORT}").parse().unwrap(),
        std::time::Duration::from_secs(1),
    )
    .is_ok()
    {
        if allow_external == Some(true) {
            return Ok(server_ready(&app, &state));
        }
        return Err(format!("Port {SERVER_PORT} is already in use. Close the other process, or connect only if you started that server."));
    }

    // Get app data directory
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app data dir: {}", e))?;

    // Ensure data directory exists
    std::fs::create_dir_all(&data_dir).map_err(|e| format!("Failed to create data dir: {}", e))?;

    println!("=================================================================");
    println!("Starting voicebox-server sidecar");
    println!("Data directory: {:?}", data_dir);
    println!("Remote mode: {}", remote.unwrap_or(false));

    // Check for ROCm backend in data directory (onedir layout: backends/rocm/)
    let rocm_binary = {
        let rocm_dir = data_dir.join("backends").join("rocm");
        let rocm_name = if cfg!(windows) {
            "voicebox-server-rocm.exe"
        } else {
            "voicebox-server-rocm"
        };
        let exe_path = rocm_dir.join(rocm_name);
        if exe_path.exists() {
            println!("Found ROCm backend at {:?}", rocm_dir);

            let app_version = app.config().version.clone().unwrap_or_default();
            let binary_version = probe_binary_version(&exe_path, &rocm_dir).await;
            let version_ok = if !app_version.is_empty()
                && binary_version.as_deref() == Some(app_version.as_str())
            {
                println!("ROCm binary version {} matches app version", app_version);
                true
            } else {
                println!(
                    "ROCm binary version mismatch: binary={}, app={}. Falling back to CPU.",
                    binary_version.as_deref().unwrap_or("<unknown>"),
                    app_version
                );
                false
            };

            if version_ok {
                Some(exe_path)
            } else {
                None
            }
        } else {
            println!("No ROCm backend found");
            None
        }
    };

    // Check for CUDA backend in data directory (onedir layout: backends/cuda/)
    let cuda_binary = {
        let cuda_dir = data_dir.join("backends").join("cuda");
        let cuda_name = if cfg!(windows) {
            "voicebox-server-cuda.exe"
        } else {
            "voicebox-server-cuda"
        };
        let exe_path = cuda_dir.join(cuda_name);
        if exe_path.exists() {
            println!("Found CUDA backend at {:?}", cuda_dir);

            // Version check: run --version from the onedir directory so
            // PyInstaller can find its support files for the fast --version path
            let app_version = app.config().version.clone().unwrap_or_default();
            let binary_version = probe_binary_version(&exe_path, &cuda_dir).await;
            let version_ok = if !app_version.is_empty()
                && binary_version.as_deref() == Some(app_version.as_str())
            {
                println!("CUDA binary version {} matches app version", app_version);
                true
            } else {
                println!(
                    "CUDA binary version mismatch: binary={}, app={}. Falling back to CPU.",
                    binary_version.as_deref().unwrap_or("<unknown>"),
                    app_version
                );
                false
            };

            if version_ok {
                Some(exe_path)
            } else {
                None
            }
        } else {
            println!("No CUDA backend found, using bundled CPU binary");
            None
        }
    };

    let mut sidecar = app
        .shell()
        .sidecar("voicebox-server")
        .map_err(|e| format!("Failed to create server command: {e}"))?;

    println!("Sidecar command created successfully");

    // Build common args
    let data_dir_str = data_dir
        .to_str()
        .ok_or_else(|| "Invalid data dir path".to_string())?
        .to_string();
    let port_str = SERVER_PORT.to_string();
    let parent_pid_str = std::process::id().to_string();
    // Resolve the custom models directory from the parameter or stored state
    let effective_models_dir = models_dir.or_else(|| state.models_dir.lock().unwrap().clone());
    if let Some(ref dir) = effective_models_dir {
        println!("Custom models directory: {}", dir);
    }

    // Respect backend override (e.g., user wants CPU even though a GPU binary
    // exists, or pinned a specific GPU variant). The in-memory value resets to
    // None on app launch, so fall back to the persisted choice on disk.
    let backend_override = {
        let in_memory = state.backend_override.lock().unwrap().clone();
        in_memory.or_else(|| read_persisted_backend_override(&data_dir))
    };

    // Honor a pinned GPU variant by ignoring the other one — but only when the
    // pinned variant is actually installed, so a stale pin to a deleted backend
    // self-heals to the default order instead of forcing CPU. With no pin, both
    // stay eligible and the launch order below prefers ROCm, then CUDA.
    let pin = backend_override.as_deref();
    let pin_cuda = pin == Some("cuda") && cuda_binary.is_some();
    let pin_rocm = pin == Some("rocm") && rocm_binary.is_some();
    let rocm_binary = if pin_cuda { None } else { rocm_binary };
    let cuda_binary = if pin_rocm { None } else { cuda_binary };

    // If ROCm binary exists, launch it from the onedir directory.
    // If CUDA binary exists, launch it from the onedir directory.
    // .current_dir() is critical: PyInstaller onedir expects all DLLs and
    // support files relative to the exe.
    let spawn_result = if backend_override.as_deref() != Some("cpu") {
        let mut gpu_spawn = None;

        if let Some(ref rocm_path) = rocm_binary {
            let rocm_dir = rocm_path.parent().unwrap();
            println!(
                "Launching ROCm backend: {:?} (cwd: {:?})",
                rocm_path, rocm_dir
            );
            let mut cmd = app.shell().command(rocm_path.to_str().unwrap());
            cmd = cmd.current_dir(rocm_dir);
            cmd = cmd.args([
                "--data-dir",
                &data_dir_str,
                "--port",
                &port_str,
                "--parent-pid",
                &parent_pid_str,
            ]);
            if bind_remote {
                cmd = cmd.args(["--host", "0.0.0.0"]);
            }
            if let Some(ref dir) = effective_models_dir {
                cmd = cmd.env("VOICEBOX_MODELS_DIR", dir);
            }
            if let Some(ref token) = effective_remote_api_token {
                cmd = cmd.env("VOICEBOX_REMOTE_API_TOKEN", token);
            }
            match cmd.spawn() {
                Ok(r) => {
                    gpu_spawn = Some(Ok(r));
                }
                Err(e) => {
                    println!("ROCm spawn failed ({}), trying CUDA/CPU fallback", e);
                }
            }
        }

        if gpu_spawn.is_none() {
            if let Some(ref cuda_path) = cuda_binary {
                let cuda_dir = cuda_path.parent().unwrap();
                println!(
                    "Launching CUDA backend: {:?} (cwd: {:?})",
                    cuda_path, cuda_dir
                );
                let mut cmd = app.shell().command(cuda_path.to_str().unwrap());
                cmd = cmd.current_dir(cuda_dir);
                cmd = cmd.args([
                    "--data-dir",
                    &data_dir_str,
                    "--port",
                    &port_str,
                    "--parent-pid",
                    &parent_pid_str,
                ]);
                if bind_remote {
                    cmd = cmd.args(["--host", "0.0.0.0"]);
                }
                if let Some(ref dir) = effective_models_dir {
                    cmd = cmd.env("VOICEBOX_MODELS_DIR", dir);
                }
                if let Some(ref token) = effective_remote_api_token {
                    cmd = cmd.env("VOICEBOX_REMOTE_API_TOKEN", token);
                }
                match cmd.spawn() {
                    Ok(r) => {
                        gpu_spawn = Some(Ok(r));
                    }
                    Err(e) => {
                        println!("CUDA spawn failed ({}), falling back to CPU", e);
                    }
                }
            }
        }

        if let Some(result) = gpu_spawn {
            result
        } else {
            // Fall back to bundled CPU sidecar
            sidecar = sidecar.args([
                "--data-dir",
                &data_dir_str,
                "--port",
                &port_str,
                "--parent-pid",
                &parent_pid_str,
            ]);
            if bind_remote {
                sidecar = sidecar.args(["--host", "0.0.0.0"]);
            }
            if let Some(ref dir) = effective_models_dir {
                sidecar = sidecar.env("VOICEBOX_MODELS_DIR", dir);
            }
            if let Some(ref token) = effective_remote_api_token {
                sidecar = sidecar.env("VOICEBOX_REMOTE_API_TOKEN", token);
            }
            println!("Spawning bundled CPU server process...");
            sidecar.spawn()
        }
    } else {
        // Override forces CPU — use bundled sidecar, GPU binary stays on disk
        println!("Backend override=cpu: using bundled CPU sidecar");
        sidecar = sidecar.args([
            "--data-dir",
            &data_dir_str,
            "--port",
            &port_str,
            "--parent-pid",
            &parent_pid_str,
        ]);
        if bind_remote {
            sidecar = sidecar.args(["--host", "0.0.0.0"]);
        }
        if let Some(ref dir) = effective_models_dir {
            sidecar = sidecar.env("VOICEBOX_MODELS_DIR", dir);
        }
        if let Some(ref token) = effective_remote_api_token {
            sidecar = sidecar.env("VOICEBOX_REMOTE_API_TOKEN", token);
        }
        sidecar.spawn()
    };

    let (mut rx, child) = spawn_result.map_err(|e| format!("Failed to spawn server: {e}"))?;

    println!("Server process spawned, waiting for ready signal...");
    println!("=================================================================");

    // Store child process and PID
    let process_pid = child.pid();
    *state.server_pid.lock().unwrap() = Some(process_pid);
    *state.child.lock().unwrap() = Some(child);
    let failed_start = scopeguard::guard((app.clone(), process_pid), |(app, pid)| {
        clear_managed_server(&app, pid, true);
    });

    // Wait for server to be ready by listening for startup log
    // PyInstaller bundles can be slow on first import, especially torch/transformers
    let timeout = tokio::time::Duration::from_secs(120);
    let start_time = tokio::time::Instant::now();
    let mut error_output = Vec::new();

    loop {
        if start_time.elapsed() > timeout {
            eprintln!("Server startup timeout after 120 seconds");
            if !error_output.is_empty() {
                eprintln!("Collected error output:");
                for line in &error_output {
                    eprintln!("  {}", line);
                }
            }

            return Err("Server startup timeout - check Console.app for detailed logs".to_string());
        }

        match tokio::time::timeout(tokio::time::Duration::from_millis(100), rx.recv()).await {
            Ok(Some(event)) => {
                match event {
                    tauri_plugin_shell::process::CommandEvent::Stdout(line) => {
                        let line_str = String::from_utf8_lossy(&line);
                        println!("Server output: {}", line_str);
                        let _ = app.emit(
                            "server-log",
                            serde_json::json!({
                                "stream": "stdout",
                                "line": line_str.trim_end(),
                            }),
                        );

                        if line_str.contains("Uvicorn running") {
                            println!("Server is ready!");
                            break;
                        }
                    }
                    tauri_plugin_shell::process::CommandEvent::Stderr(line) => {
                        let line_str = String::from_utf8_lossy(&line).to_string();
                        eprintln!("Server: {}", line_str);
                        let _ = app.emit(
                            "server-log",
                            serde_json::json!({
                                "stream": "stderr",
                                "line": line_str.trim_end(),
                            }),
                        );

                        // Collect error lines for debugging
                        if line_str.contains("ERROR")
                            || line_str.contains("Error")
                            || line_str.contains("Failed")
                        {
                            error_output.push(line_str.clone());
                        }

                        // Uvicorn logs to stderr, so check there too
                        if line_str.contains("Uvicorn running") {
                            println!("Server is ready!");
                            break;
                        }
                    }
                    tauri_plugin_shell::process::CommandEvent::Terminated(_) => {
                        clear_managed_server(&app, process_pid, false);
                        return Err("Server process ended during startup".into());
                    }
                    tauri_plugin_shell::process::CommandEvent::Error(error) => {
                        return Err(format!("Server process error: {error}"));
                    }
                    _ => {}
                }
            }
            Ok(None) => return Err("Server process ended unexpectedly".to_string()),
            Err(_) => {
                // Timeout on this recv, continue loop
                continue;
            }
        }
    }

    let server_url = server_ready(&app, &state);
    scopeguard::ScopeGuard::into_inner(failed_start);

    // Spawn task to continue reading output and emit to frontend
    let app_handle = app.clone();
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                tauri_plugin_shell::process::CommandEvent::Stdout(line) => {
                    let line_str = String::from_utf8_lossy(&line);
                    println!("Server: {}", line_str);
                    let _ = app_handle.emit(
                        "server-log",
                        serde_json::json!({
                            "stream": "stdout",
                            "line": line_str.trim_end(),
                        }),
                    );
                }
                tauri_plugin_shell::process::CommandEvent::Stderr(line) => {
                    let line_str = String::from_utf8_lossy(&line);
                    eprintln!("Server error: {}", line_str);
                    let _ = app_handle.emit(
                        "server-log",
                        serde_json::json!({
                            "stream": "stderr",
                            "line": line_str.trim_end(),
                        }),
                    );
                }
                tauri_plugin_shell::process::CommandEvent::Terminated(_) => {
                    clear_managed_server(&app_handle, process_pid, false);
                    return;
                }
                tauri_plugin_shell::process::CommandEvent::Error(error) => {
                    eprintln!("Server process error: {error}");
                    clear_managed_server(&app_handle, process_pid, true);
                    return;
                }
                _ => {}
            }
        }
        clear_managed_server(&app_handle, process_pid, false);
    });

    Ok(server_url)
}

#[command]
async fn stop_server(state: State<'_, ServerState>) -> Result<(), String> {
    if let Some(monitor) = state.speak_monitor.lock().unwrap().take() {
        monitor.abort();
    }
    let pid = state.server_pid.lock().unwrap().take();
    let _child = state.child.lock().unwrap().take();

    if let Some(pid) = pid {
        println!("stop_server: Stopping server with PID: {}", pid);

        #[cfg(unix)]
        {
            use std::process::Command;
            // Kill process group with SIGTERM first
            let _ = Command::new("kill")
                .args(["-TERM", "--", &format!("-{}", pid)])
                .output();

            // Brief wait then force kill
            std::thread::sleep(std::time::Duration::from_millis(100));

            let _ = Command::new("kill")
                .args(["-9", "--", &format!("-{}", pid)])
                .output();
            let _ = Command::new("kill").args(["-9", &pid.to_string()]).output();

            println!("stop_server: Process group kill completed");
        }

        #[cfg(windows)]
        {
            // Send graceful shutdown via HTTP — the server's parent-pid watchdog
            // will also handle cleanup if this app process exits.
            println!("Sending graceful shutdown via HTTP...");
            let client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(2))
                .build()
                .unwrap();

            let _ = client
                .post(&format!("http://127.0.0.1:{}/shutdown", SERVER_PORT))
                .send();

            println!("Shutdown request sent (server watchdog will handle cleanup)");
        }
    }

    Ok(())
}

#[command]
async fn restart_server(
    app: tauri::AppHandle,
    state: State<'_, ServerState>,
    models_dir: Option<String>,
) -> Result<String, String> {
    println!("restart_server: stopping current server...");

    // Update stored models_dir: empty string means reset to default, non-empty means set
    if let Some(ref dir) = models_dir {
        if dir.is_empty() {
            *state.models_dir.lock().unwrap() = None;
        } else {
            *state.models_dir.lock().unwrap() = Some(dir.clone());
        }
    }

    // Stop the current server
    stop_server(state.clone()).await?;

    // Wait for port to be released
    println!("restart_server: waiting for port release...");
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Start server again (will auto-detect GPU binary and use stored models_dir)
    println!("restart_server: starting server...");
    start_server(app, state.clone(), None, None, None, None).await
}

#[command]
fn set_keep_server_running(state: State<'_, ServerState>, keep_running: bool) {
    println!("set_keep_server_running called with: {}", keep_running);
    *state.keep_running_on_close.lock().unwrap() = keep_running;
}

#[command]
fn set_backend_override(
    app: tauri::AppHandle,
    state: State<'_, ServerState>,
    backend: Option<String>,
) {
    println!("set_backend_override called with: {:?}", backend);
    if let Ok(data_dir) = app.path().app_data_dir() {
        write_persisted_backend_override(&data_dir, backend.as_deref());
    }
    *state.backend_override.lock().unwrap() = backend;
}

/// Open a filesystem directory, never a renderer-supplied URL or executable.
#[command]
fn open_directory(
    app: tauri::AppHandle,
    window: tauri::WebviewWindow,
    state: State<ServerState>,
    path: String,
    connection_id: String,
) -> Result<(), String> {
    if window.label() != "main" {
        return Err("Only the main window can open server directories".into());
    }
    let connection = state.client_connection.lock().map_err(|e| e.to_string())?;
    let connection = connection
        .as_ref()
        .ok_or("No server connection is available")?;
    if connection.connection_id != connection_id
        || !local_directory_request(&connection.server_url, &path)
    {
        return Err("Only local server directories can be opened".into());
    }
    let directory = std::path::Path::new(&path)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if !directory.is_dir() {
        return Err("Only directories can be opened".into());
    }
    app.shell()
        .open(directory.to_string_lossy().as_ref(), None)
        .map_err(|e| e.to_string())
}

fn local_directory_request(server_url: &str, path: &str) -> bool {
    // Reject UNC/device paths before canonicalization can contact a network share.
    if path.starts_with("//")
        || path.starts_with("\\\\")
        || !std::path::Path::new(path).is_absolute()
    {
        return false;
    }
    let Ok(url) = tauri::Url::parse(server_url) else {
        return false;
    };
    matches!(url.scheme(), "http" | "https")
        && matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"))
        && url.port_or_known_default() == Some(SERVER_PORT)
        && url.username().is_empty()
        && url.password().is_none()
}

fn trusted_webview_origin(value: &str) -> bool {
    let Ok(url) = tauri::Url::parse(value) else {
        return false;
    };
    if !url.username().is_empty() || url.password().is_some() {
        return false;
    }
    match (url.scheme(), url.host_str(), url.port()) {
        ("tauri", Some("localhost"), None) => true,
        ("http" | "https", Some("tauri.localhost"), None) => true,
        ("http", Some("localhost" | "127.0.0.1"), Some(5173)) => cfg!(debug_assertions),
        _ => false,
    }
}

#[command]
async fn start_system_audio_capture(
    state: State<'_, audio_capture::AudioCaptureState>,
    max_duration_secs: u32,
) -> Result<(), String> {
    audio_capture::start_capture(&state, max_duration_secs).await
}

#[command]
async fn stop_system_audio_capture(
    state: State<'_, audio_capture::AudioCaptureState>,
) -> Result<String, String> {
    audio_capture::stop_capture(&state).await
}

#[command]
fn is_system_audio_supported() -> bool {
    audio_capture::is_supported()
}

#[command]
fn list_audio_output_devices(
    state: State<'_, audio_output::AudioOutputState>,
) -> Result<Vec<audio_output::AudioOutputDevice>, String> {
    state.list_output_devices()
}

#[command]
async fn play_audio_to_devices(
    state: State<'_, audio_output::AudioOutputState>,
    audio_data: Vec<u8>,
    device_ids: Vec<String>,
) -> Result<(), String> {
    state.play_audio_to_devices(audio_data, device_ids).await
}

#[command]
fn stop_audio_playback(state: State<'_, audio_output::AudioOutputState>) -> Result<(), String> {
    state.stop_all_playback()
}

/// Identifier of the Voicebox app itself — used to short-circuit auto-paste
/// when the user fires a chord while focus was inside one of our own
/// windows. Paste into Voicebox-internal targets is step 6 territory and
/// goes through a different (JS-side) injection path.
///
/// Value matches what `focus_capture::capture_focus` writes into
/// `FocusSnapshot::bundle_id` on the current platform — reverse-DNS bundle
/// id on macOS, lowercased exe basename on Windows/Linux.
#[cfg(target_os = "macos")]
const VOICEBOX_BUNDLE_ID: &str = "sh.voicebox.app";
#[cfg(target_os = "windows")]
const VOICEBOX_BUNDLE_ID: &str = "voicebox.exe";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const VOICEBOX_BUNDLE_ID: &str = "voicebox";

/// Milliseconds to wait between activating the target app and firing the
/// synthetic ⌘V, giving AppKit time to finish re-ordering windows and
/// restoring its last-focused field.
const POST_ACTIVATE_SETTLE_MS: u64 = 120;

/// Milliseconds the staged text lives on the clipboard after the paste
/// keystroke, before we restore the user's original clipboard contents.
/// Too short and slow apps haven't consumed the paste yet; too long and
/// the user sees our text if they look at their clipboard manager.
const PASTE_CONSUME_MS: u64 = 400;

/// Reports whether the process currently has macOS Accessibility trust.
/// Used by the settings UI and the paste debug harness to decide whether
/// synthetic key events will actually land.
#[command]
fn check_accessibility_permission() -> bool {
    accessibility::is_trusted()
}

/// Reports whether the process can observe global keyboard events. Read by
/// the Captures settings UI to surface a "missing — open Settings" hint
/// beside the hotkey toggle. No prompt side-effect.
#[command]
fn check_input_monitoring_permission() -> bool {
    input_monitoring::is_trusted()
}

/// Holds the lazily-spawned global hotkey monitor. The monitor is `None`
/// until the user opts in via the Captures settings toggle — that opt-in is
/// what triggers the macOS Input Monitoring TCC prompt, so a fresh-install
/// user who never enables the hotkey never sees the prompt.
///
/// Disabling the hotkey clears the monitor's internal `ChordMatcher` so
/// keytap's event tap is released while Tauri still owns this `HotkeyState`
/// for the rest of the process. A subsequent enable re-arms without
/// re-prompting for the Input Monitoring permission.
#[cfg(desktop)]
#[derive(Default)]
pub struct HotkeyState {
    monitor: Mutex<Option<hotkey_monitor::HotkeyMonitor>>,
}

#[cfg(desktop)]
fn build_chord_bindings(
    push_to_talk: &[String],
    toggle_to_talk: &[String],
) -> Result<hotkey_monitor::Bindings, String> {
    use hotkey_monitor::{Bindings, ChordAction};
    use keytap::Key;
    use std::collections::HashSet;

    fn build_chord(name: &str, names: &[String]) -> Result<HashSet<Key>, String> {
        if names.is_empty() {
            return Err(format!("{name} chord must have at least one key"));
        }
        let mut chord = HashSet::new();
        for raw in names {
            let key = key_codes::key_from_str(raw)
                .ok_or_else(|| format!("Unsupported key in {name} chord: {raw}"))?;
            chord.insert(key);
        }
        Ok(chord)
    }

    let push_chord = build_chord("push-to-talk", push_to_talk)?;
    let toggle_chord = build_chord("toggle-to-talk", toggle_to_talk)?;

    let mut bindings = Bindings::new();
    bindings.insert(ChordAction::PushToTalk, push_chord);
    bindings.insert(ChordAction::ToggleToTalk, toggle_chord);
    Ok(bindings)
}

/// Spawn the global hotkey monitor on first call; subsequent calls just push
/// the new bindings into the existing monitor. Idempotent on purpose — the
/// frontend invokes this both at startup (when `capture_settings.hotkey_enabled`
/// is true) and from the settings toggle.
///
/// On macOS this is the call that triggers the "Voicebox would like to receive
/// keystrokes from any application" TCC prompt, since keytap's `Tap` creates
/// the CGEventTap inside `HotkeyMonitor::spawn`.
#[cfg(desktop)]
#[command]
fn enable_hotkey(
    app: tauri::AppHandle,
    state: State<'_, HotkeyState>,
    push_to_talk: Vec<String>,
    toggle_to_talk: Vec<String>,
) -> Result<(), String> {
    let bindings = build_chord_bindings(&push_to_talk, &toggle_to_talk)?;

    // Fire the Input Monitoring TCC prompt explicitly from the user's
    // toggle click, before keytap's Tap would do it implicitly via
    // CGEventTap creation. Two reasons: (1) the prompt timing becomes
    // deterministic — it appears in response to a click instead of as a
    // mysterious side-effect of "the app started"; (2) on subsequent
    // launches we can short-circuit the spawn entirely if the user
    // revoked the grant, instead of relying on the tap silently failing.
    // The call returns the current grant state; we ignore it because
    // keytap surfaces its own error via stderr, and the settings UI
    // polls `check_input_monitoring_permission` separately.
    let _ = input_monitoring::request();

    // The dictate pill webview must exist before the first chord fires so it
    // can subscribe to `dictate:start`. Build it here (idempotent — Tauri
    // returns the existing window when one with this label already exists).
    if app.get_webview_window(DICTATE_WINDOW_LABEL).is_none() {
        if let Err(e) = build_dictate_window(&app) {
            eprintln!("Failed to build dictate window: {}", e);
        }
    }

    let mut slot = state.monitor.lock().map_err(|e| e.to_string())?;
    match slot.as_mut() {
        Some(monitor) => monitor.update_bindings(bindings),
        None => {
            *slot = Some(hotkey_monitor::HotkeyMonitor::spawn(app, bindings));
        }
    }
    Ok(())
}

/// Quiet the global hotkey. Tears down the `ChordMatcher` (which stops
/// keytap's chord worker and closes the OS event tap) but keeps the
/// `HotkeyMonitor` handle around so a subsequent `enable_hotkey` re-arms
/// without re-prompting for Input Monitoring permission.
#[cfg(desktop)]
#[command]
fn disable_hotkey(state: State<'_, HotkeyState>) -> Result<(), String> {
    let mut slot = state.monitor.lock().map_err(|e| e.to_string())?;
    if let Some(monitor) = slot.as_mut() {
        monitor.update_bindings(hotkey_monitor::Bindings::new());
    }
    Ok(())
}

/// Push a new chord configuration into the running `HotkeyMonitor`. Called
/// by the chord-picker UI when the user edits the chord. No-ops when the
/// monitor isn't spawned — the picker is gated behind the enable toggle, so
/// this can only happen if the frontend races; the next `enable_hotkey` will
/// pick up the saved chords.
///
/// Returns an error when a key name doesn't map to a `keytap::Key`, so the
/// picker UI can surface "this key isn't supported" instead of silently
/// dropping it from the chord.
#[cfg(desktop)]
#[command]
fn update_chord_bindings(
    state: State<'_, HotkeyState>,
    push_to_talk: Vec<String>,
    toggle_to_talk: Vec<String>,
) -> Result<(), String> {
    let bindings = build_chord_bindings(&push_to_talk, &toggle_to_talk)?;
    let mut slot = state.monitor.lock().map_err(|e| e.to_string())?;
    if let Some(monitor) = slot.as_mut() {
        monitor.update_bindings(bindings);
    }
    Ok(())
}

/// Open the Privacy & Security → Accessibility pane in System Settings so
/// the user can grant the permission. The URL scheme is stable across
/// macOS 10.14–15; no-op on other platforms.
#[command]
fn open_accessibility_settings(app: tauri::AppHandle) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility";
        app.shell()
            .open(url, None)
            .map_err(|e| format!("Failed to open Accessibility settings: {e}"))?;
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = app;
        Err("Accessibility settings pane is only implemented on macOS".into())
    }
}

/// Open the Privacy & Security → Input Monitoring pane in System Settings.
/// Used by the Captures settings UI when the toggle is on but the grant
/// is missing, so the user can flip the system toggle without hunting.
#[command]
fn open_input_monitoring_settings(app: tauri::AppHandle) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let url = "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent";
        app.shell()
            .open(url, None)
            .map_err(|e| format!("Failed to open Input Monitoring settings: {e}"))?;
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = app;
        Err("Input Monitoring settings pane is only implemented on macOS".into())
    }
}

/// Deliver `text` into the UI that had focus when the chord fired.
///
/// Pipeline: activate the captured PID → settle → save the user's
/// clipboard → write `text` → fire ⌘V → wait for the target to consume it
/// → conditionally restore the original clipboard.
///
/// The restore is conditional on `NSPasteboard.changeCount` (or the
/// Windows sequence number) matching the value captured right after
/// `write_text`: if something else wrote to the clipboard during the
/// paste-consume window — the user's own ⌘C in the target app, a
/// clipboard history tool (Paste, Pastebot, Maccy), Universal Clipboard
/// sync, 1Password inserting a secret — their newer content takes
/// priority over our snapshot and is preserved. A
/// [`clipboard::current_change_count`] read failure is treated the same
/// way: unknown state is safer than an unconditional overwrite.
///
/// `send_paste` failure is isolated from the restore decision: we always
/// attempt the conditional restore before propagating the paste error,
/// so a failed `CGEventPost` / `SendInput` never leaves the user's
/// clipboard stuck on the transcript.
///
/// Skips (returns `false`) without touching anything when:
/// - `focus.bundle_id` is Voicebox itself — step 6 will inject directly
///   into our own webview; pasting would just double-insert or miss the
///   real target.
/// - Accessibility is not trusted — `CGEventPost` would silently drop the
///   keystroke, leaving the user's clipboard clobbered with nothing to
///   show for it.
///
/// Returns `true` when the paste sequence completed end-to-end.
#[command]
async fn paste_final_text(
    text: String,
    focus: u64,
    targets: State<'_, focus_capture::PendingFocusTargets>,
) -> Result<bool, String> {
    let focus = targets.take(focus)?;
    if text.len() > 1024 * 1024 {
        return Err("Dictation text exceeds 1 MiB".into());
    }
    if focus.bundle_id.as_deref() == Some(VOICEBOX_BUNDLE_ID) {
        return Ok(false);
    }
    if !accessibility::is_trusted() {
        return Err(
            "Accessibility permission required for auto-paste. Open System Settings → Privacy & Security → Accessibility and enable Voicebox."
                .into(),
        );
    }

    focus_capture::activate_pid(focus.pid)?;
    tokio::time::sleep(std::time::Duration::from_millis(POST_ACTIVATE_SETTLE_MS)).await;

    let current = focus_capture::capture_focus()?;
    if current.pid != focus.pid || current.bundle_id != focus.bundle_id {
        return Err("The original dictation target is no longer available".into());
    }
    let snapshot = clipboard::save_clipboard()?;
    let after_write = clipboard::write_text(&text)?;

    let paste_result = synthetic_keys::send_paste();
    tokio::time::sleep(std::time::Duration::from_millis(PASTE_CONSUME_MS)).await;

    let safe_to_restore = matches!(
        clipboard::current_change_count(),
        Ok(current) if current == after_write
    );
    if safe_to_restore {
        clipboard::restore_clipboard(&snapshot)?;
    } else {
        eprintln!(
            "[voicebox] clipboard mutated during paste window — skipping restore to preserve newer content"
        );
    }

    paste_result?;
    Ok(true)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .manage(focus_capture::PendingFocusTargets::default())
        .manage(ServerState {
            client_connection: Mutex::new(None),
            child: Mutex::new(None),
            start_lock: tokio::sync::Mutex::new(()),
            speak_monitor: Mutex::new(None),
            server_pid: Mutex::new(None),
            keep_running_on_close: Mutex::new(false),
            models_dir: Mutex::new(None),
            remote_mode: Mutex::new(false),
            remote_api_token: Mutex::new(None),
            backend_override: Mutex::new(None),
        })
        .manage(audio_capture::AudioCaptureState::new())
        .manage(audio_output::AudioOutputState::new())
        .setup(|app| {
            #[cfg(desktop)]
            {
                app.handle()
                    .plugin(tauri_plugin_updater::Builder::new().build())?;
                app.handle().plugin(tauri_plugin_process::init())?;

                // Resolve the active keyboard layout's V keycode now, on
                // the main thread, and register an observer for layout
                // changes. The synthetic-paste hot path then only reads an
                // atomic. See keyboard_layout.rs for why this matters
                // (Cmd+V is matched by translated character, not keycode,
                // so QWERTY keycode 9 produces Cmd+. on Dvorak).
                keyboard_layout::init();

                // HotkeyMonitor is spawned lazily via the `enable_hotkey`
                // command — see HotkeyState. The hidden dictate webview is
                // safe to build up front because it does not create the global
                // keyboard tap or trigger the macOS Input Monitoring prompt.
                app.manage(HotkeyState::default());

                // The frontend emits `dictate:hide` whenever the pill cycle
                // finishes (rest-fade → hidden). `hide()` alone has been
                // unreliable for transparent always-on-top windows on macOS
                // — the NSWindow lingers as an invisible click target that
                // steals focus to the Voicebox app when the user clicks
                // where it used to be. Park the window off-screen and mark
                // it click-through as well, so even if `hide()` no-ops the
                // user sees and interacts with nothing.
                let handle_for_hide = app.handle().clone();
                app.handle().listen("dictate:hide", move |_event| {
                    if let Some(window) = handle_for_hide.get_webview_window(DICTATE_WINDOW_LABEL) {
                        // Skip on Linux: aborts if the window was never realized
                        // (see show_dictate_window).
                        #[cfg(not(target_os = "linux"))]
                        let _ = window.set_ignore_cursor_events(true);
                        let _ = window.set_position(PhysicalPosition::new(-10_000, -10_000));
                        let _ = window.hide();
                    }
                });

                // Agent-initiated speech (voicebox.speak over MCP or POST /speak)
                // pops the pill up so the user can see what's coming out of their
                // machine. The `dictate:show` listener is kept for any frontend
                // caller that wants to force-surface the pill directly, but the
                // primary source is `speak_monitor` below — Rust subscribes to
                // the backend /events/speak SSE stream so the pill surfaces even
                // when no JS window is active.
                let handle_for_show = app.handle().clone();
                app.handle().listen("dictate:show", move |_event| {
                    show_dictate_window(&handle_for_show);
                });

                ensure_dictate_window(app.handle());
            }

            // Hide title bar icon on Windows
            #[cfg(windows)]
            {
                use windows::Win32::Foundation::HWND;
                use windows::Win32::UI::WindowsAndMessaging::{
                    SetClassLongPtrW, GCLP_HICON, GCLP_HICONSM,
                };

                if let Some((_, window)) = app.webview_windows().iter().next() {
                    if let Ok(hwnd) = window.hwnd() {
                        let hwnd = HWND(hwnd.0);
                        unsafe {
                            // Set both small and regular icons to NULL to hide the title bar icon
                            SetClassLongPtrW(hwnd, GCLP_HICON, 0);
                            SetClassLongPtrW(hwnd, GCLP_HICONSM, 0);
                        }
                    }
                }
            }

            // Enable microphone access on Linux (WebKitGTK denies getUserMedia by default)
            #[cfg(target_os = "linux")]
            {
                use tauri::Manager;
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.with_webview(|webview| {
                        use webkit2gtk::glib::ObjectExt;
                        use webkit2gtk::{PermissionRequestExt, SettingsExt, WebViewExt};
                        let wk_webview = webview.inner();

                        // Enable media stream support in WebKitGTK settings
                        if let Some(settings) = WebViewExt::settings(&wk_webview) {
                            settings.set_enable_media_stream(true);
                        }

                        // Auto-grant UserMediaPermissionRequest (microphone access)
                        // Only for trusted local origins (Tauri dev server or custom protocol)
                        wk_webview.connect_permission_request(
                            move |webview, request: &webkit2gtk::PermissionRequest| {
                                if request.is::<webkit2gtk::UserMediaPermissionRequest>() {
                                    let uri = WebViewExt::uri(webview).unwrap_or_default();
                                    let is_trusted = trusted_webview_origin(&uri);
                                    if is_trusted {
                                        request.allow();
                                        return true;
                                    }
                                    request.deny();
                                    return true;
                                }
                                false
                            },
                        );
                    });
                }
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            set_client_connection,
            get_client_connection,
            start_server,
            open_directory,
            stop_server,
            restart_server,
            set_keep_server_running,
            set_backend_override,
            start_system_audio_capture,
            stop_system_audio_capture,
            is_system_audio_supported,
            list_audio_output_devices,
            play_audio_to_devices,
            stop_audio_playback,
            check_accessibility_permission,
            check_input_monitoring_permission,
            open_accessibility_settings,
            open_input_monitoring_settings,
            paste_final_text,
            enable_hotkey,
            disable_hotkey,
            update_chord_bindings
        ])
        .on_window_event({
            let closing = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            move |window, event| {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    // If we're already in the close flow, let it proceed
                    if closing.load(std::sync::atomic::Ordering::SeqCst) {
                        return;
                    }
                    closing.store(true, std::sync::atomic::Ordering::SeqCst);

                    // Prevent automatic close so frontend can clean up
                    api.prevent_close();

                    // Emit event to frontend to check setting and stop server if needed
                    let app_handle = window.app_handle();

                    if let Err(e) = app_handle.emit("window-close-requested", ()) {
                        eprintln!("Failed to emit window-close-requested event: {}", e);
                        window.close().ok();
                        return;
                    }

                    // Set up listener for frontend response
                    let window_for_close = window.clone();
                    let closing_for_timeout = closing.clone();
                    let (tx, mut rx) = mpsc::unbounded_channel::<()>();

                    let listener_id = window.listen("window-close-allowed", move |_| {
                        let _ = tx.send(());
                    });

                    tauri::async_runtime::spawn(async move {
                        tokio::select! {
                            _ = rx.recv() => {
                                window_for_close.close().ok();
                            }
                            _ = tokio::time::sleep(tokio::time::Duration::from_secs(5)) => {
                                eprintln!("Window close timeout, closing anyway");
                                window_for_close.close().ok();
                            }
                        }
                        window_for_close.unlisten(listener_id);
                        closing_for_timeout.store(false, std::sync::atomic::Ordering::SeqCst);
                    });
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            let _ = &app; // used on unix
            match &event {
                RunEvent::Exit => {
                    let state = app.state::<ServerState>();
                    let keep_running = *state.keep_running_on_close.lock().unwrap();
                    let has_pid = state.server_pid.lock().unwrap().is_some();
                    println!(
                        "RunEvent::Exit — keep_running={}, has_pid={}",
                        keep_running, has_pid
                    );

                    if keep_running {
                        // Tell the server to disable its watchdog so it survives
                        // after this process exits.
                        println!("Keep server running: disabling watchdog...");

                        // Write a sentinel file as a reliable fallback. On Windows
                        // the HTTP request below can race with process exit, leaving
                        // the watchdog unaware it should stay alive. The sentinel
                        // file is checked during the watchdog grace period.
                        let data_dir = app.path().app_data_dir().unwrap_or_default();
                        let sentinel = data_dir.join(".keep-running");
                        if let Err(e) = std::fs::write(&sentinel, b"1") {
                            eprintln!("Failed to write keep-running sentinel: {}", e);
                        } else {
                            println!("Wrote keep-running sentinel to {:?}", sentinel);
                        }

                        let client = reqwest::blocking::Client::builder()
                            .timeout(std::time::Duration::from_secs(2))
                            .build()
                            .unwrap();
                        match client
                            .post(&format!(
                                "http://127.0.0.1:{}/watchdog/disable",
                                SERVER_PORT
                            ))
                            .send()
                        {
                            Ok(resp) => println!("Watchdog disable response: {}", resp.status()),
                            Err(e) => eprintln!("Failed to disable watchdog: {}", e),
                        }
                    } else {
                        // Server will self-terminate via parent-pid watchdog when
                        // this process exits. On Unix, also send SIGTERM for
                        // immediate cleanup.
                        println!("RunEvent::Exit - server will self-terminate via watchdog");

                        #[cfg(unix)]
                        {
                            if let Some(pid) = state.server_pid.lock().unwrap().take() {
                                use std::process::Command;
                                let _ = Command::new("kill")
                                    .args(["-TERM", "--", &format!("-{}", pid)])
                                    .output();
                                std::thread::sleep(std::time::Duration::from_millis(100));
                                let _ = Command::new("kill")
                                    .args(["-9", "--", &format!("-{}", pid)])
                                    .output();
                                let _ =
                                    Command::new("kill").args(["-9", &pid.to_string()]).output();
                            }
                        }
                    }
                }
                RunEvent::ExitRequested { api, .. } => {
                    println!("RunEvent::ExitRequested received");
                    // Don't prevent exit, just log it
                    let _ = api;
                }
                _ => {}
            }
        });
}

fn main() {
    run();
}

#[cfg(test)]
mod security_tests {
    use super::{local_directory_request, trusted_webview_origin};

    #[test]
    fn directory_open_rejects_remote_and_network_paths_before_filesystem_access() {
        let local_path = std::env::temp_dir();
        let local_path = local_path.to_str().unwrap();
        for server in [
            "http://127.0.0.1:17493",
            "http://localhost:17493",
            "http://[::1]:17493",
        ] {
            assert!(local_directory_request(server, local_path), "{server}");
        }
        for server in [
            "https://remote.example",
            "http://localhost:8000",
            "file://localhost:17493",
            "http://user@localhost:17493",
        ] {
            assert!(!local_directory_request(server, local_path), "{server}");
        }
        for path in [
            "//host/share",
            "\\\\host\\share",
            "\\\\?\\UNC\\host\\share",
            "relative/path",
            "https://example.com",
        ] {
            assert!(
                !local_directory_request("http://127.0.0.1:17493", path),
                "{path}"
            );
        }
    }

    #[test]
    fn media_permission_requires_exact_app_origin() {
        assert!(trusted_webview_origin("tauri://localhost/index.html"));
        assert!(trusted_webview_origin("http://tauri.localhost/index.html"));
        assert_eq!(
            trusted_webview_origin("http://localhost:5173"),
            cfg!(debug_assertions)
        );
        for origin in [
            "http://localhost.evil.com",
            "http://127.0.0.1.evil.com",
            "http://localhost:8000",
            "tauri://evil",
            "https://tauri.localhost.evil.com",
            "http://user@localhost:5173",
        ] {
            assert!(!trusted_webview_origin(origin), "{origin}");
        }
    }
}
