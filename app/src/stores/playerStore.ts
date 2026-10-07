import { create } from 'zustand';
import { MAX_BUFFERED_AUDIO_BYTES } from '@/lib/constants/audio';

interface PlayerState {
  audioUrl: string | null;
  audioSourceKind: 'server' | 'local';
  audioId: string | null;
  profileId: string | null;
  title: string | null;
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  isLooping: boolean;
  shouldRestart: boolean;
  shouldAutoPlay: boolean;
  onFinish: (() => void) | null;

  setAudio: (url: string, id: string, profileId: string | null, title?: string) => void;
  setAudioWithAutoPlay: (url: string, id: string, profileId: string | null, title?: string) => void;
  setLocalAudioWithAutoPlay: (blob: Blob, id: string, title?: string) => void;
  setIsPlaying: (playing: boolean) => void;
  setCurrentTime: (time: number) => void;
  setDuration: (duration: number) => void;
  setVolume: (volume: number) => void;
  toggleLoop: () => void;
  restartCurrentAudio: () => void;
  clearRestartFlag: () => void;
  clearAutoPlayFlag: () => void;
  setOnFinish: (callback: (() => void) | null) => void;
  reset: () => void;
}

function releaseLocalAudio(state: PlayerState): void {
  if (state.audioSourceKind === 'local' && state.audioUrl) {
    URL.revokeObjectURL(state.audioUrl);
  }
}

export const usePlayerStore = create<PlayerState>((set, get) => {
  const setAudio = (
    url: string,
    id: string,
    profileId: string | null,
    title: string | undefined,
    shouldAutoPlay: boolean,
    audioSourceKind: PlayerState['audioSourceKind'] = 'server',
  ) => {
    releaseLocalAudio(get());
    set({
      audioUrl: url,
      audioSourceKind,
      audioId: id,
      profileId: profileId || null,
      title: title || null,
      currentTime: 0,
      isPlaying: false,
      shouldRestart: false,
      shouldAutoPlay,
    });
  };

  return {
    audioUrl: null,
    audioSourceKind: 'server',
    audioId: null,
    profileId: null,
    title: null,
    isPlaying: false,
    currentTime: 0,
    duration: 0,
    volume: 1,
    isLooping: false,
    shouldRestart: false,
    shouldAutoPlay: false,
    onFinish: null,

    setAudio: (url, id, profileId, title) => setAudio(url, id, profileId, title, false),
    setAudioWithAutoPlay: (url, id, profileId, title) => setAudio(url, id, profileId, title, true),
    // The store owns these URLs until replacement/reset, independently of the
    // preview component or waveform mounting. Callers provide bytes, not URLs.
    setLocalAudioWithAutoPlay: (blob, id, title) => {
      if (blob.size > MAX_BUFFERED_AUDIO_BYTES) {
        throw new Error('Audio exceeds the allowed playback size');
      }
      setAudio(URL.createObjectURL(blob), id, null, title, true, 'local');
    },
    setIsPlaying: (playing) => set({ isPlaying: playing }),
    setCurrentTime: (time) => set({ currentTime: time }),
    setDuration: (duration) => set({ duration }),
    setVolume: (volume) => set({ volume }),
    toggleLoop: () => set((state) => ({ isLooping: !state.isLooping })),
    restartCurrentAudio: () => set({ shouldRestart: true }),
    clearRestartFlag: () => set({ shouldRestart: false }),
    clearAutoPlayFlag: () => set({ shouldAutoPlay: false }),
    setOnFinish: (callback) => set({ onFinish: callback }),
    reset: () => {
      releaseLocalAudio(get());
      set({
        audioUrl: null,
        audioSourceKind: 'server',
        audioId: null,
        profileId: null,
        title: null,
        isPlaying: false,
        currentTime: 0,
        duration: 0,
        isLooping: false,
        shouldRestart: false,
        shouldAutoPlay: false,
        onFinish: null,
      });
    },
  };
});
