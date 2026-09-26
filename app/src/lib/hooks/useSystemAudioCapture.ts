import { useCallback, useEffect, useRef, useState } from 'react';
import { usePlatform } from '@/platform/PlatformContext';

interface UseSystemAudioCaptureOptions {
  maxDurationSeconds?: number;
  onRecordingComplete?: (blob: Blob, duration?: number) => void;
}

interface CaptureSession {
  cancelled: boolean;
  startedAt: number | null;
  stopping: boolean;
  onComplete: UseSystemAudioCaptureOptions['onRecordingComplete'];
}

/**
 * Hook for native system audio capture using Tauri commands.
 * Uses ScreenCaptureKit on macOS and WASAPI loopback on Windows.
 */
export function useSystemAudioCapture({
  maxDurationSeconds = 29,
  onRecordingComplete,
}: UseSystemAudioCaptureOptions = {}) {
  const platform = usePlatform();
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isSupported, setIsSupported] = useState(false);
  const timerRef = useRef<number | null>(null);
  const stopRecordingRef = useRef<(() => Promise<void>) | null>(null);
  const sessionRef = useRef<CaptureSession | null>(null);

  // Check if system audio capture is supported
  useEffect(() => {
    let isActive = true;

    void platform.audio
      .isSystemAudioSupported()
      .then((supported) => {
        if (isActive) {
          setIsSupported(supported);
        }
      })
      .catch(() => {
        if (isActive) {
          setIsSupported(false);
        }
      });

    return () => {
      isActive = false;
    };
  }, [platform]);

  const startRecording = useCallback(async () => {
    // A pending native stop still owns the capture device.
    if (sessionRef.current) return;
    if (!platform.metadata.isTauri) {
      const errorMsg = 'System audio capture is only available in the desktop app.';
      setError(errorMsg);
      return;
    }

    if (!isSupported) {
      const errorMsg = 'System audio capture is not supported on this platform.';
      setError(errorMsg);
      return;
    }

    const session: CaptureSession = {
      cancelled: false,
      startedAt: null,
      stopping: false,
      onComplete: onRecordingComplete,
    };
    sessionRef.current = session;
    try {
      setError(null);
      setDuration(0);

      // Start native capture
      await platform.audio.startSystemAudioCapture(maxDurationSeconds);
      if (session.cancelled) {
        await platform.audio.stopSystemAudioCapture();
        if (sessionRef.current === session) sessionRef.current = null;
        return;
      }

      setIsRecording(true);
      session.startedAt = Date.now();

      // Start timer
      timerRef.current = window.setInterval(() => {
        if (session.startedAt !== null) {
          const elapsed = (Date.now() - session.startedAt) / 1000;
          setDuration(elapsed);

          // Auto-stop at max duration
          if (elapsed >= maxDurationSeconds && stopRecordingRef.current) {
            void stopRecordingRef.current();
          }
        }
      }, 100);
    } catch (err) {
      if (sessionRef.current === session) sessionRef.current = null;
      if (session.cancelled) {
        console.error('Error releasing cancelled system audio capture:', err);
        return;
      }
      const errorMessage =
        err instanceof Error
          ? err.message
          : 'Failed to start system audio capture. Please check permissions.';
      setError(errorMessage);
      setIsRecording(false);
    }
  }, [maxDurationSeconds, isSupported, onRecordingComplete, platform]);

  const stopRecording = useCallback(async () => {
    const session = sessionRef.current;
    if (!session || session.startedAt === null || session.stopping) {
      return;
    }

    session.stopping = true;
    const recordedDuration = (Date.now() - session.startedAt) / 1000;
    try {
      if (!session.cancelled) setIsRecording(false);

      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      // Stop capture and get Blob
      const blob = await platform.audio.stopSystemAudioCapture();

      if (!session.cancelled) session.onComplete?.(blob, recordedDuration);
    } catch (err) {
      if (session.cancelled) {
        console.error('Error stopping cancelled system audio capture:', err);
        return;
      }
      const errorMessage =
        err instanceof Error ? err.message : 'Failed to stop system audio capture.';
      setError(errorMessage);
    } finally {
      if (sessionRef.current === session) sessionRef.current = null;
    }
  }, [platform]);

  // Store stopRecording in ref for use in timer
  useEffect(() => {
    stopRecordingRef.current = stopRecording;
  }, [stopRecording]);

  const cancelRecording = useCallback(async () => {
    if (sessionRef.current) sessionRef.current.cancelled = true;

    setIsRecording(false);
    setDuration(0);

    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    await stopRecording();
  }, [stopRecording]);

  // Cleanup on unmount only
  useEffect(() => {
    return () => {
      if (sessionRef.current) sessionRef.current.cancelled = true;
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      void stopRecording();
    };
  }, [stopRecording]);

  return {
    isRecording,
    duration,
    error,
    isSupported,
    startRecording,
    stopRecording,
    cancelRecording,
  };
}
