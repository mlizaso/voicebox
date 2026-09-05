use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleFormat, StreamConfig};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, serde::Serialize)]
pub struct AudioOutputDevice {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

const MAX_AUDIO_BYTES: usize = 100 * 1024 * 1024;
const MAX_PCM_SAMPLES: usize = 25_000_000;

fn start_output_stream(
    stopped: &AtomicBool,
    start: impl FnOnce() -> Result<(), String>,
    on_started: impl FnOnce(),
) -> Result<(), String> {
    if stopped.load(Ordering::Acquire) {
        return Err("Playback cancelled".into());
    }
    start()?;
    if stopped.load(Ordering::Acquire) {
        return Err("Playback cancelled".into());
    }
    on_started();
    Ok(())
}

// CPAL exposes names, not stable hardware IDs. Preserve existing IDs for the
// first occurrence and disambiguate duplicate or normalized-equal names.
fn device_id(name: &str, used: &mut HashSet<String>) -> String {
    let base = format!("device_{}", name.replace(' ', "_").to_lowercase());
    let mut id = base.clone();
    let mut suffix = 2;
    while !used.insert(id.clone()) {
        id = format!("{base}__{suffix}");
        suffix += 1;
    }
    id
}

pub struct AudioOutputState {
    host: Host,
    stop_flag: Mutex<Arc<AtomicBool>>,
}

impl AudioOutputState {
    pub fn new() -> Self {
        Self {
            host: cpal::default_host(),
            stop_flag: Mutex::new(Arc::new(AtomicBool::new(false))),
        }
    }

    pub fn stop_all_playback(&self) -> Result<(), String> {
        eprintln!("stop_all_playback: Setting stop flag");
        self.stop_flag
            .lock()
            .unwrap()
            .store(true, Ordering::Relaxed);
        eprintln!("stop_all_playback: Stop flag set - active streams will output silence");
        Ok(())
    }

    pub fn list_output_devices(&self) -> Result<Vec<AudioOutputDevice>, String> {
        let devices = self
            .host
            .output_devices()
            .map_err(|e| format!("Failed to enumerate output devices: {}", e))?;

        let default_device = self.host.default_output_device();

        let mut result = Vec::new();
        let mut used_ids = HashSet::new();
        for device in devices {
            let name = device
                .name()
                .map_err(|e| format!("Failed to get device name: {}", e))?;

            // Generate a stable ID from the device name (cpal doesn't provide stable IDs)
            let id = device_id(&name, &mut used_ids);

            let is_default = default_device
                .as_ref()
                .map(|d| d.name().unwrap_or_default() == name)
                .unwrap_or(false);

            result.push(AudioOutputDevice {
                id,
                name,
                is_default,
            });
        }

        Ok(result)
    }

    pub async fn play_audio_to_devices(
        &self,
        audio_data: Vec<u8>,
        device_ids: Vec<String>,
    ) -> Result<(), String> {
        eprintln!(
            "play_audio_to_devices called with {} bytes, {} device IDs",
            audio_data.len(),
            device_ids.len()
        );
        eprintln!("Requested device IDs: {:?}", device_ids);

        // Decode audio file (assuming WAV format)
        eprintln!("Decoding audio data...");
        if device_ids.is_empty() || device_ids.len() > 8 {
            return Err("Select between 1 and 8 output devices".into());
        }
        let stop_flag = Arc::new(AtomicBool::new(false));
        {
            let mut previous = self.stop_flag.lock().unwrap();
            previous.store(true, Ordering::Relaxed);
            *previous = stop_flag.clone();
        }
        let (samples, sample_rate, channels) = Self::decode_wav(audio_data, MAX_PCM_SAMPLES)?;
        if samples.len().saturating_mul(device_ids.len()) > MAX_PCM_SAMPLES {
            return Err("Audio is too large for the selected output devices".into());
        }
        eprintln!(
            "Audio decoded: {} samples, {}Hz, {} channels",
            samples.len(),
            sample_rate,
            channels
        );

        // Find devices by ID
        eprintln!("Enumerating output devices...");
        let mut used_ids = HashSet::new();
        let devices: Vec<Device> = self
            .host
            .output_devices()
            .map_err(|e| format!("Failed to enumerate devices: {}", e))?
            .filter_map(|device| {
                let name = device.name().ok()?;
                let id = device_id(&name, &mut used_ids);
                eprintln!("Found device: {} (id: {})", name, id);
                if device_ids.contains(&id) {
                    eprintln!("  -> Matched! Will play to this device");
                    Some(device)
                } else {
                    None
                }
            })
            .collect();

        if devices.is_empty() {
            eprintln!("ERROR: No matching devices found");
            return Err("No matching devices found".to_string());
        }

        eprintln!("Playing to {} device(s)", devices.len());

        let mut starts = Vec::new();
        for device in devices {
            let audio = samples.clone();
            let stop = stop_flag.clone();
            let (started, receiver) = tokio::sync::oneshot::channel();
            starts.push(receiver);
            tokio::task::spawn_blocking(move || {
                let mut started = Some(started);
                let result = Self::play_to_device(
                    &device,
                    audio,
                    sample_rate,
                    channels,
                    stop.clone(),
                    || {
                        if let Some(tx) = started.take() {
                            let _ = tx.send(Ok(()));
                        }
                    },
                );
                if let Err(error) = result {
                    stop.store(true, Ordering::Relaxed);
                    if let Some(tx) = started.take() {
                        let _ = tx.send(Err(error.clone()));
                    }
                    eprintln!("Audio output failed: {error}");
                }
            });
        }
        // Report readiness as soon as the streams start, while the workers
        // retain their streams until completion or explicit stop.
        for start in starts {
            match tokio::time::timeout(std::time::Duration::from_secs(10), start).await {
                Ok(Ok(Ok(()))) => {}
                result => {
                    stop_flag.store(true, Ordering::Relaxed);
                    return Err(match result {
                        Ok(Ok(Err(error))) => error,
                        _ => "Audio output did not start".into(),
                    });
                }
            }
        }
        Ok(())
    }

    fn decode_wav(data: Vec<u8>, max_samples: usize) -> Result<(Vec<f32>, u32, u16), String> {
        if data.len() > MAX_AUDIO_BYTES {
            return Err("Audio input exceeds 100 MiB".into());
        }
        use symphonia::core::formats::FormatOptions;
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;

        eprintln!(
            "decode_wav: Creating MediaSourceStream from {} bytes",
            data.len()
        );
        let mss = MediaSourceStream::new(Box::new(std::io::Cursor::new(data)), Default::default());

        eprintln!("decode_wav: Probing audio format...");
        let mut format = symphonia::default::get_probe()
            .format(
                &Default::default(),
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| {
                eprintln!("decode_wav: Failed to probe audio: {}", e);
                format!("Failed to probe audio: {}", e)
            })?
            .format;

        eprintln!("decode_wav: Audio format probed successfully");

        eprintln!("decode_wav: Finding audio track...");
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or_else(|| {
                eprintln!("decode_wav: No audio track found");
                "No audio track found".to_string()
            })?;

        let sample_rate = track.codec_params.sample_rate.ok_or_else(|| {
            eprintln!("decode_wav: No sample rate found in track");
            "No sample rate found".to_string()
        })?;

        let channels = track
            .codec_params
            .channels
            .ok_or_else(|| {
                eprintln!("decode_wav: No channels found in track");
                "No channels found".to_string()
            })?
            .count() as u16;
        if !(1..=192_000).contains(&sample_rate) || !(1..=8).contains(&channels) {
            return Err("Unsupported audio sample rate or channel count".into());
        }
        if track
            .codec_params
            .n_frames
            .is_some_and(|frames| frames > (max_samples / channels as usize) as u64)
        {
            return Err("Decoded audio exceeds the playback memory limit".into());
        }

        eprintln!(
            "decode_wav: Track info - sample_rate: {}, channels: {}",
            sample_rate, channels
        );

        eprintln!("decode_wav: Creating decoder...");
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &Default::default())
            .map_err(|e| {
                eprintln!("decode_wav: Failed to create decoder: {}", e);
                format!("Failed to create decoder: {}", e)
            })?;

        eprintln!("decode_wav: Decoder created successfully");

        let mut samples = Vec::new();
        let mut packet_count = 0;
        eprintln!("decode_wav: Starting packet decoding loop...");
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(e) => {
                    eprintln!("decode_wav: End of stream or error: {:?}", e);
                    break;
                }
            };

            packet_count += 1;
            let decoded = decoder.decode(&packet).map_err(|e| {
                eprintln!("decode_wav: Decode error on packet {}: {}", packet_count, e);
                format!("Decode error: {}", e)
            })?;

            // Convert to f32 samples by matching on the buffer type
            use symphonia::core::audio::{AudioBufferRef, Signal};
            use symphonia::core::conv::FromSample;

            let spec = *decoded.spec();
            let num_channels = spec.channels.count();
            let num_frames = decoded.frames();
            if spec.rate != sample_rate || num_channels != channels as usize {
                return Err("Audio format changes during playback are unsupported".into());
            }
            if num_frames
                .saturating_mul(num_channels)
                .saturating_add(samples.len())
                > max_samples
            {
                return Err("Decoded audio exceeds the playback memory limit".into());
            }

            eprintln!(
                "decode_wav: Packet {} - {} frames, {} channels",
                packet_count, num_frames, num_channels
            );

            // Interleave samples from all channels
            for frame_idx in 0..num_frames {
                for ch in 0..num_channels {
                    let sample_f32 = match &decoded {
                        AudioBufferRef::U8(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::U16(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::U24(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::U32(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::S8(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::S16(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::S24(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::S32(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::F32(buf) => buf.chan(ch)[frame_idx],
                        AudioBufferRef::F64(buf) => buf.chan(ch)[frame_idx] as f32,
                    };
                    samples.push(sample_f32);
                }
            }
        }

        eprintln!(
            "decode_wav: Decoded {} packets, total {} samples",
            packet_count,
            samples.len()
        );
        eprintln!(
            "decode_wav: Returning sample_rate={}, channels={}",
            sample_rate, channels
        );
        Ok((samples, sample_rate, channels))
    }

    fn play_to_device(
        device: &Device,
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        stop_flag: Arc<AtomicBool>,
        on_started: impl FnOnce(),
    ) -> Result<(), String> {
        let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
        eprintln!(
            "play_to_device: Starting playback to device: {}",
            device_name
        );
        eprintln!(
            "play_to_device: Input - {} samples, {}Hz, {} channels",
            samples.len(),
            sample_rate,
            channels
        );

        let config = device
            .default_output_config()
            .map_err(|e| format!("Failed to get default config: {}", e))?;

        // Prepare samples for the device's format
        let device_sample_rate = config.sample_rate().0;
        let device_channels = config.channels();
        let device_sample_format = config.sample_format();
        if !(1..=192_000).contains(&device_sample_rate) || !(1..=8).contains(&device_channels) {
            return Err("Unsupported output sample rate or channel count".into());
        }
        let output_frames = (samples.len() as u64 / channels as u64)
            .saturating_mul(device_sample_rate as u64)
            .div_ceil(sample_rate as u64);
        if output_frames.saturating_mul(channels.max(device_channels) as u64)
            > MAX_PCM_SAMPLES as u64
        {
            return Err("Resampled audio exceeds the playback memory limit".into());
        }

        eprintln!(
            "play_to_device: Device config - {}Hz, {} channels, format: {:?}",
            device_sample_rate, device_channels, device_sample_format
        );

        // Resample if needed (simple linear interpolation for now)
        let resampled = if device_sample_rate != sample_rate {
            eprintln!(
                "play_to_device: Resampling from {}Hz to {}Hz",
                sample_rate, device_sample_rate
            );
            let result = Self::resample(&samples, sample_rate, device_sample_rate);
            eprintln!(
                "play_to_device: Resampled {} samples to {} samples",
                samples.len(),
                result.len()
            );
            result
        } else {
            eprintln!("play_to_device: No resampling needed");
            samples
        };

        // Interleave/convert channels if needed
        eprintln!(
            "play_to_device: Interleaving channels from {} to {} channels",
            channels, device_channels
        );
        let interleaved = Self::interleave_channels(&resampled, channels, device_channels);
        eprintln!(
            "play_to_device: Interleaved to {} samples",
            interleaved.len()
        );

        // Create shared buffer for playback
        let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(interleaved));
        let position = Arc::new(AtomicUsize::new(0));
        let buffer_clone = buffer.clone();
        let position_clone = position.clone();

        let err_fn = |err| eprintln!("Playback error: {}", err);

        let stream_config = StreamConfig {
            channels: device_channels,
            sample_rate: cpal::SampleRate(device_sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let stop_flag_clone = stop_flag.clone();
        let stream = match config.sample_format() {
            SampleFormat::F32 => {
                let buffer = buffer_clone.clone();
                let pos = position_clone.clone();
                device
                    .build_output_stream(
                        &stream_config,
                        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                            // Check stop flag - if set, output silence
                            if stop_flag_clone.load(Ordering::Relaxed) {
                                for sample in data.iter_mut() {
                                    *sample = 0.0;
                                }
                                return;
                            }

                            let mut idx = pos.load(Ordering::Relaxed);
                            let buf = buffer.lock().unwrap();
                            for sample in data.iter_mut() {
                                if idx < buf.len() {
                                    *sample = buf[idx];
                                    idx += 1;
                                } else {
                                    *sample = 0.0;
                                }
                            }
                            pos.store(idx, Ordering::Relaxed);
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("Failed to build stream: {}", e))?
            }
            SampleFormat::I16 => {
                let buffer = buffer_clone.clone();
                let pos = position_clone.clone();
                device
                    .build_output_stream(
                        &stream_config,
                        move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                            // Check stop flag - if set, output silence
                            if stop_flag_clone.load(Ordering::Relaxed) {
                                for sample in data.iter_mut() {
                                    *sample = 0;
                                }
                                return;
                            }

                            let mut idx = pos.load(Ordering::Relaxed);
                            let buf = buffer.lock().unwrap();
                            for sample in data.iter_mut() {
                                if idx < buf.len() {
                                    *sample = (buf[idx] * 32767.0) as i16;
                                    idx += 1;
                                } else {
                                    *sample = 0;
                                }
                            }
                            pos.store(idx, Ordering::Relaxed);
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("Failed to build stream: {}", e))?
            }
            SampleFormat::U16 => {
                let buffer = buffer_clone.clone();
                let pos = position_clone.clone();
                device
                    .build_output_stream(
                        &stream_config,
                        move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                            // Check stop flag - if set, output silence
                            if stop_flag_clone.load(Ordering::Relaxed) {
                                for sample in data.iter_mut() {
                                    *sample = 32768;
                                }
                                return;
                            }

                            let mut idx = pos.load(Ordering::Relaxed);
                            let buf = buffer.lock().unwrap();
                            for sample in data.iter_mut() {
                                if idx < buf.len() {
                                    *sample = ((buf[idx] + 1.0) * 32767.5) as u16;
                                    idx += 1;
                                } else {
                                    *sample = 32768;
                                }
                            }
                            pos.store(idx, Ordering::Relaxed);
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("Failed to build stream: {}", e))?
            }
            _ => return Err("Unsupported sample format".to_string()),
        };

        eprintln!("play_to_device: Starting stream playback...");
        start_output_stream(
            &stop_flag,
            || {
                stream
                    .play()
                    .map_err(|e| format!("Failed to play stream: {e}"))
            },
            on_started,
        )?;
        eprintln!("play_to_device: Stream started successfully");

        // Keep the stream alive until playback finishes.
        // Previously the stream was dropped immediately on function return,
        // causing silent playback (cpal stops output when its Stream is dropped).
        let total_samples = { buffer.lock().unwrap().len() };
        loop {
            let pos = position.load(std::sync::atomic::Ordering::Relaxed);
            if pos >= total_samples || stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // stream is dropped here, after audio has finished playing
        drop(stream);
        eprintln!("play_to_device: Function completed successfully");
        Ok(())
    }

    fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        if from_rate == to_rate {
            return samples.to_vec();
        }

        let ratio = to_rate as f64 / from_rate as f64;
        let new_len = (samples.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = (i as f64 / ratio) as usize;
            if src_idx < samples.len() {
                resampled.push(samples[src_idx]);
            } else {
                resampled.push(0.0);
            }
        }

        resampled
    }

    fn interleave_channels(samples: &[f32], src_channels: u16, dst_channels: u16) -> Vec<f32> {
        if src_channels == dst_channels {
            return samples.to_vec();
        }

        let mut interleaved = Vec::new();
        let samples_per_channel = samples.len() / src_channels as usize;

        for i in 0..samples_per_channel {
            for ch in 0..dst_channels {
                let src_ch = if ch < src_channels {
                    ch
                } else {
                    src_channels - 1
                };
                let idx = (i * src_channels as usize) + src_ch as usize;
                if idx < samples.len() {
                    interleaved.push(samples[idx]);
                } else {
                    interleaved.push(0.0);
                }
            }
        }

        interleaved
    }
}

impl Default for AudioOutputState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod security_tests {
    use super::*;

    #[tokio::test]
    async fn cancelled_output_never_reports_readiness() {
        for cancel_during_start in [false, true] {
            let stopped = AtomicBool::new(!cancel_during_start);
            let (ready, receiver) = tokio::sync::oneshot::channel();
            let result = start_output_stream(
                &stopped,
                || {
                    assert!(cancel_during_start);
                    stopped.store(true, Ordering::Release);
                    Ok(())
                },
                || {
                    let _ = ready.send(());
                },
            );
            assert_eq!(result.unwrap_err(), "Playback cancelled");
            assert!(receiver.await.is_err());
        }
    }

    #[test]
    fn normalized_duplicate_device_names_keep_distinct_ids() {
        let mut used = HashSet::new();
        let ids: Vec<_> = ["USB Audio", "USB Audio", "usb_audio", "usb_audio__2"]
            .map(|name| device_id(name, &mut used))
            .into_iter()
            .collect();
        assert_eq!(ids[0], "device_usb_audio");
        assert_eq!(ids.iter().collect::<HashSet<_>>().len(), 4);
    }

    fn wav(channels: u16, rate: u32) -> Vec<u8> {
        let mut data = Vec::new();
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(std::io::Cursor::new(&mut data), spec).unwrap();
        for _ in 0..channels * 4 {
            writer.write_sample(42_i16).unwrap();
        }
        writer.finalize().unwrap();
        data
    }

    #[test]
    fn decode_checks_declared_and_actual_pcm_bounds() {
        let (samples, rate, channels) = AudioOutputState::decode_wav(wav(1, 24000), 4).unwrap();
        assert_eq!((samples.len(), rate, channels), (4, 24000, 1));
        assert!(AudioOutputState::decode_wav(wav(1, 24000), 3)
            .unwrap_err()
            .contains("memory limit"));
        assert!(AudioOutputState::decode_wav(wav(9, 24000), 100).is_err());
        assert!(AudioOutputState::decode_wav(wav(1, 192001), 100).is_err());
        assert!(AudioOutputState::decode_wav(vec![0; 44], 100).is_err());
    }
}
