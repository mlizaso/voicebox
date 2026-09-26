import { useEffect, useState } from 'react';
import { getAudioDuration } from '@/lib/utils/audio';

interface AudioDurationResult {
  file: File | null;
  duration: number | null;
  error: unknown;
  isLoading: boolean;
}

/** Metadata belongs to the selected file, even when an older decode finishes later. */
export function useAudioDuration(file: File | undefined) {
  const [result, setResult] = useState<AudioDurationResult>({
    file: null,
    duration: null,
    error: null,
    isLoading: false,
  });

  useEffect(() => {
    if (!file) return;
    let active = true;
    void getAudioDuration(file).then(
      (duration) => {
        if (active) setResult({ file, duration, error: null, isLoading: false });
      },
      (error: unknown) => {
        if (active) setResult({ file, duration: null, error, isLoading: false });
      },
    );
    return () => {
      active = false;
    };
  }, [file]);

  return result.file === (file ?? null)
    ? result
    : { file: file ?? null, duration: null, error: null, isLoading: !!file };
}
