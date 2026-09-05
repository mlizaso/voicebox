import { useCallback, useEffect, useRef, useState } from 'react';
import { convertToWav } from '@/lib/utils/audio';
import { usePlatform } from '@/platform/PlatformContext';

interface UseAudioRecordingOptions {
  maxDurationSeconds?: number;
  onRecordingComplete?: (blob: Blob, duration?: number) => void;
}

export function useAudioRecording({
  maxDurationSeconds,
  onRecordingComplete,
}: UseAudioRecordingOptions = {}) {
  const platform = usePlatform();
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const sessionRef = useRef<{ cancelled: boolean } | null>(null);

  const startingRef = useRef(false);
  const startRecording = useCallback(
    async (onComplete = onRecordingComplete) => {
      if (startingRef.current || mediaRecorderRef.current?.state === 'recording') return;
      startingRef.current = true;
      const session = { cancelled: false };
      sessionRef.current = session;
      try {
        setError(null);
        setDuration(0);

        // Check if getUserMedia is available
        // In Tauri, navigator.mediaDevices might not be available immediately
        if (typeof navigator === 'undefined') {
          const errorMsg =
            'Navigator API is not available. This might be a Tauri configuration issue.';
          setError(errorMsg);
          throw new Error(errorMsg);
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          // Try waiting a bit for Tauri webview to initialize
          await new Promise((resolve) => setTimeout(resolve, 100));

          if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.error('MediaDevices check:', {
              hasNavigator: typeof navigator !== 'undefined',
              hasMediaDevices: !!navigator?.mediaDevices,
              hasGetUserMedia: !!navigator?.mediaDevices?.getUserMedia,
              isTauri: platform.metadata.isTauri,
            });

            const errorMsg = platform.metadata.isTauri
              ? 'Microphone access is not available. Please ensure:\n1. The app has microphone permissions in System Settings (macOS: System Settings > Privacy & Security > Microphone)\n2. You restart the app after granting permissions\n3. You are using Tauri v2 with a webview that supports getUserMedia'
              : 'Microphone access is not available. Please ensure you are using a secure context (HTTPS or localhost) and that your browser has microphone permissions enabled.';
            setError(errorMsg);
            throw new Error(errorMsg);
          }
        }

        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });

        if (session.cancelled) {
          stream.getTracks().forEach((track) => {
            track.stop();
          });
          return;
        }

        streamRef.current = stream;
        const chunks: Blob[] = [];

        // Create MediaRecorder with preferred MIME type
        const options: MediaRecorderOptions = {
          mimeType: 'audio/webm;codecs=opus',
        };

        // Fallback to default if webm not supported
        if (!MediaRecorder.isTypeSupported(options.mimeType!)) {
          delete options.mimeType;
        }

        const mediaRecorder = new MediaRecorder(stream, options);
        mediaRecorderRef.current = mediaRecorder;
        const recordingStartedAt = Date.now();

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            chunks.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          // These values belong to this recording, even if a new one has started.
          const wasCancelled = session.cancelled;
          const recordedDuration = (Date.now() - recordingStartedAt) / 1000;

          const webmBlob = new Blob(chunks, { type: mediaRecorder.mimeType });

          // Stop all tracks now that we have the data
          stream.getTracks().forEach((track) => {
            if (track.readyState !== 'ended') track.stop();
          });
          if (streamRef.current === stream) streamRef.current = null;

          // Don't fire completion callback if the recording was cancelled
          if (wasCancelled) return;

          // Convert to WAV format to avoid needing ffmpeg on backend
          try {
            const wavBlob = await convertToWav(webmBlob);
            if (!session.cancelled) onComplete?.(wavBlob, recordedDuration);
          } catch (err) {
            console.error('Error converting audio to WAV:', err);
            // Fallback to original blob if conversion fails
            if (!session.cancelled) onComplete?.(webmBlob, recordedDuration);
          }
        };

        mediaRecorder.onerror = (event) => {
          setError('Recording error occurred');
          console.error('MediaRecorder error:', event);
        };

        // WebKit's MediaRecorder drops the WebM EBML header from chunks when
        // started with a timeslice, so concatenated blobs fail to parse in
        // both AudioContext and ffmpeg. Starting with no timeslice produces
        // exactly one dataavailable on stop() with a valid container.
        mediaRecorder.start();
        setIsRecording(true);
        startTimeRef.current = Date.now();

        // Start timer
        timerRef.current = window.setInterval(() => {
          if (startTimeRef.current) {
            const elapsed = (Date.now() - startTimeRef.current) / 1000;
            setDuration(elapsed);

            // Auto-stop at max duration when the caller opts in — dictation
            // sessions pass undefined and run until the user releases the
            // chord or hits stop; voice-clone sample recorders pass 29s to
            // keep reference clips short.
            if (maxDurationSeconds !== undefined && elapsed >= maxDurationSeconds) {
              if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
                mediaRecorderRef.current.stop();
                setIsRecording(false);
                if (timerRef.current !== null) {
                  clearInterval(timerRef.current);
                  timerRef.current = null;
                }
              }
            }
          }
        }, 100);
      } catch (err) {
        streamRef.current?.getTracks().forEach((track) => {
          if (track.readyState !== 'ended') track.stop();
        });
        streamRef.current = null;
        const errorMessage =
          err instanceof Error
            ? err.message
            : 'Failed to access microphone. Please check permissions.';
        setError(errorMessage);
        setIsRecording(false);
      } finally {
        startingRef.current = false;
      }
    },
    [maxDurationSeconds, onRecordingComplete, platform.metadata.isTauri],
  );

  const stopRecording = useCallback(() => {
    if (startingRef.current && sessionRef.current) sessionRef.current.cancelled = true;
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  }, []);

  const cancelRecording = useCallback(() => {
    if (sessionRef.current) sessionRef.current.cancelled = true;
    if (mediaRecorderRef.current) {
      if (mediaRecorderRef.current.state !== 'inactive') mediaRecorderRef.current.stop();
      setIsRecording(false);
      setDuration(0);
    }

    // Stop all tracks
    streamRef.current?.getTracks().forEach((track) => {
      if (track.readyState !== 'ended') track.stop();
    });
    streamRef.current = null;

    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (sessionRef.current) sessionRef.current.cancelled = true;
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
      }
      streamRef.current?.getTracks().forEach((track) => {
        if (track.readyState !== 'ended') track.stop();
      });
    };
  }, []);

  return {
    isRecording,
    audioStream: isRecording ? streamRef.current : null,
    duration,
    error,
    startRecording: () => startRecording(),
    startRecordingWithCallback: startRecording,
    stopRecording,
    cancelRecording,
  };
}
