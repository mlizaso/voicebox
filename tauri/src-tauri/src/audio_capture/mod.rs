#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "linux")]
use linux as platform;
#[cfg(target_os = "macos")]
use macos as platform;
#[cfg(target_os = "windows")]
use windows as platform;

pub use platform::is_supported;

pub const MAX_CAPTURE_SECONDS: u32 = 30;
// Bound callback storage even if a device or timer misbehaves.
pub const MAX_CAPTURE_SAMPLES: usize = 8_000_000;

#[cfg(any(target_os = "windows", test))]
fn capture_packet_bytes(
    frames: usize,
    channels: usize,
    bytes_per_sample: usize,
) -> Result<usize, String> {
    if !(1..=8).contains(&channels) || !(1..=8).contains(&bytes_per_sample) {
        return Err("Unsupported capture format".into());
    }
    let samples = frames
        .checked_mul(channels)
        .filter(|count| *count <= MAX_CAPTURE_SAMPLES)
        .ok_or("Capture packet exceeds the memory limit")?;
    samples
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| "Capture packet size overflow".into())
}

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(target_os = "macos")]
use screencapturekit::stream::sc_stream::SCStream;

pub struct AudioCaptureState {
    operation: tokio::sync::Mutex<Option<AudioCaptureSession>>,
}

// Callbacks and timers retain only their own recording's buffers and stop signal.
pub struct AudioCaptureSession {
    pub finished: Arc<AtomicBool>,
    pub samples: Arc<Mutex<Vec<f32>>>,
    pub sample_rate: Arc<Mutex<u32>>,
    pub channels: Arc<Mutex<u16>>,
    pub stop_tx: Arc<Mutex<Option<tokio::sync::mpsc::Sender<()>>>>,
    pub error: Arc<Mutex<Option<String>>>,
    #[cfg(target_os = "macos")]
    pub stream: Arc<Mutex<Option<SCStream>>>,
}

impl AudioCaptureState {
    pub fn new() -> Self {
        Self {
            operation: tokio::sync::Mutex::new(None),
        }
    }
}

impl AudioCaptureSession {
    fn new() -> Self {
        Self {
            finished: Arc::new(AtomicBool::new(false)),
            samples: Arc::new(Mutex::new(Vec::new())),
            sample_rate: Arc::new(Mutex::new(44100)),
            channels: Arc::new(Mutex::new(2)),
            stop_tx: Arc::new(Mutex::new(None)),
            error: Arc::new(Mutex::new(None)),
            #[cfg(target_os = "macos")]
            stream: Arc::new(Mutex::new(None)),
        }
    }
}

pub async fn start_capture(
    state: &AudioCaptureState,
    max_duration_secs: u32,
) -> Result<(), String> {
    if !(1..=MAX_CAPTURE_SECONDS).contains(&max_duration_secs) {
        return Err(format!(
            "Capture duration must be between 1 and {MAX_CAPTURE_SECONDS} seconds"
        ));
    }
    let mut active = state.operation.lock().await;
    if active
        .as_ref()
        .is_some_and(|session| !session.finished.load(Ordering::Acquire))
    {
        return Err(
            "A system audio capture is already active; stop it before starting another".into(),
        );
    }
    let session = AudioCaptureSession::new();
    platform::start_capture(&session, max_duration_secs).await?;
    *active = Some(session);
    Ok(())
}

pub async fn stop_capture(state: &AudioCaptureState) -> Result<String, String> {
    let mut active = state.operation.lock().await;
    let session = active.take().ok_or("No system audio capture is active")?;
    platform::stop_capture(&session).await
}

#[cfg(any(target_os = "linux", target_os = "windows", test))]
async fn wait_for_start(
    session: &AudioCaptureSession,
    started: tokio::sync::oneshot::Receiver<()>,
) -> Result<(), String> {
    if matches!(
        tokio::time::timeout(std::time::Duration::from_secs(10), started).await,
        Ok(Ok(()))
    ) {
        return Ok(());
    }
    if let Some(tx) = session.stop_tx.lock().unwrap().take() {
        let _ = tx.try_send(());
    }
    Err(session
        .error
        .lock()
        .unwrap()
        .clone()
        .unwrap_or_else(|| "System audio capture failed to start".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_device_packets_are_rejected_before_allocation() {
        assert_eq!(capture_packet_bytes(480, 2, 4).unwrap(), 3840);
        for (frames, channels, bytes) in [
            (usize::MAX, 2, 4),
            (MAX_CAPTURE_SAMPLES + 1, 1, 4),
            (1, 0, 4),
            (1, 2, usize::MAX),
        ] {
            assert!(capture_packet_bytes(frames, channels, bytes).is_err());
        }
    }

    #[tokio::test]
    async fn invalid_or_duplicate_capture_never_opens_a_device() {
        let state = AudioCaptureState::new();
        for duration in [0, MAX_CAPTURE_SECONDS + 1, u32::MAX] {
            assert!(start_capture(&state, duration)
                .await
                .unwrap_err()
                .contains("duration"));
        }
        *state.operation.lock().await = Some(AudioCaptureSession::new());
        assert!(start_capture(&state, 29)
            .await
            .unwrap_err()
            .contains("already active"));
    }

    #[tokio::test]
    async fn failed_start_reports_error_and_stops_its_worker() {
        let session = AudioCaptureSession::new();
        *session.error.lock().unwrap() = Some("No monitor device".into());
        let (stop, mut stopped) = tokio::sync::mpsc::channel(1);
        *session.stop_tx.lock().unwrap() = Some(stop);
        let (started, receiver) = tokio::sync::oneshot::channel();
        drop(started);
        assert_eq!(
            wait_for_start(&session, receiver).await.unwrap_err(),
            "No monitor device"
        );
        assert_eq!(stopped.recv().await, Some(()));
    }

    #[test]
    fn late_callbacks_and_timeout_cannot_change_the_next_capture() {
        let previous = AudioCaptureSession::new();
        let current = AudioCaptureSession::new();
        let old_callback = previous.samples.clone();
        old_callback.lock().unwrap().push(0.5);
        previous.finished.store(true, Ordering::Release);
        assert!(current.samples.lock().unwrap().is_empty());
        assert!(!current.finished.load(Ordering::Acquire));
    }
}
