import { useCallback, useEffect, useRef, useState } from 'react';
import { useToast } from '@/components/ui/use-toast';

interface PreviewSession {
  file: File;
  audio: HTMLAudioElement;
  url: string;
  playing: boolean;
  attempt: number;
}

export function useAudioPlayer() {
  const [isPlaying, setIsPlaying] = useState(false);
  const sessionRef = useRef<PreviewSession | null>(null);
  const { toast } = useToast();

  const release = useCallback(() => {
    const session = sessionRef.current;
    sessionRef.current = null;
    if (!session) return;
    session.audio.pause();
    session.audio.removeAttribute('src');
    session.audio.load();
    URL.revokeObjectURL(session.url);
  }, []);

  useEffect(() => release, [release]);

  const cleanup = useCallback(() => {
    release();
    setIsPlaying(false);
  }, [release]);

  const playPause = (file: File | null | undefined) => {
    if (!file) return;
    let session = sessionRef.current;
    if (session?.file !== file) {
      cleanup();
      const url = URL.createObjectURL(file);
      session = { file, audio: new Audio(url), url, playing: false, attempt: 0 };
      sessionRef.current = session;
      const current = session;
      current.audio.addEventListener('ended', () => {
        if (sessionRef.current === current) cleanup();
      });
      current.audio.addEventListener('error', () => {
        if (sessionRef.current !== current) return;
        cleanup();
        toast({
          title: 'Playback error',
          description: 'Failed to play audio file',
          variant: 'destructive',
        });
      });
    }

    const current = session;
    const attempt = ++current.attempt;
    if (current.playing) {
      current.playing = false;
      current.audio.pause();
      setIsPlaying(false);
      return;
    }

    current.playing = true;
    setIsPlaying(true);
    void current.audio.play().catch(() => {
      if (sessionRef.current !== current || current.attempt !== attempt) return;
      cleanup();
      toast({
        title: 'Playback error',
        description: 'Failed to play audio file',
        variant: 'destructive',
      });
    });
  };

  return {
    isPlaying,
    playPause,
    cleanup,
  };
}
