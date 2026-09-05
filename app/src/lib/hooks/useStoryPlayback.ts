import { useCallback, useEffect, useRef } from 'react';
import { toast } from '@/components/ui/use-toast';
import { loadAudioSource, releaseAudioSource } from '@/lib/api/audioSource';
import { apiClient } from '@/lib/api/client';
import type { StoryItemDetail } from '@/lib/api/types';
import { useStoryStore } from '@/stores/storyStore';

interface ActiveSource {
  source: MediaElementAudioSourceNode;
  audio: HTMLAudioElement;
  request: AbortController;
  clipGain: GainNode;
  itemId: string;
  generationId: string;
  startTimeMs: number;
  endTimeMs: number;
}

/**
 * Hook for managing timecode-based story playback using Web Audio API.
 * Supports multiple simultaneous audio sources for overlapping clips on different tracks.
 * Uses the AudioContext clock to synchronize the timeline and per-clip gains.
 */
export function useStoryPlayback() {
  const isPlaying = useStoryStore((state) => state.isPlaying);
  const playbackItems = useStoryStore((state) => state.playbackItems);
  const playbackStartContextTime = useStoryStore((state) => state.playbackStartContextTime);
  const playbackStartStoryTime = useStoryStore((state) => state.playbackStartStoryTime);
  const setPlaybackTiming = useStoryStore((state) => state.setPlaybackTiming);

  // AudioContext instance (created once)
  const audioContextRef = useRef<AudioContext | null>(null);
  // Master gain for volume control
  const masterGainRef = useRef<GainNode | null>(null);
  // Currently playing AudioBufferSourceNodes by item.id (unique per clip)
  const activeSourcesRef = useRef<Map<string, ActiveSource>>(new Map());
  // Animation frame for syncing visual playhead
  const animationFrameRef = useRef<number | null>(null);

  // Get or create AudioContext and audio graph
  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
      console.log(
        '[StoryPlayback] Created AudioContext, sample rate:',
        audioContextRef.current.sampleRate,
      );

      // Create master gain node for volume control
      masterGainRef.current = audioContextRef.current.createGain();
      masterGainRef.current.gain.value = 1;
      masterGainRef.current.connect(audioContextRef.current.destination);
    }
    // Resume context if suspended (browser autoplay policy)
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume().catch(() => {
        // Ignore resume errors
      });
    }
    return audioContextRef.current;
  }, []);

  // Stop a source by item id
  const stopSource = useCallback((itemId: string) => {
    const activeSource = activeSourcesRef.current.get(itemId);
    if (activeSource) {
      activeSource.request.abort();
      releaseAudioSource(activeSource.audio.src);
      activeSource.audio.onended = null;
      activeSource.audio.onloadedmetadata = null;
      activeSource.audio.onerror = null;
      activeSource.audio.pause();
      activeSource.audio.removeAttribute('src');
      activeSource.audio.load();
      try {
        activeSource.source.disconnect();
      } catch {
        // already disconnected
      }
      try {
        activeSource.clipGain.disconnect();
      } catch {
        // already disconnected
      }
      activeSourcesRef.current.delete(itemId);
    }
  }, []);

  // HTML media elements stream and seek large clips without materializing
  // their full PCM data in the renderer. Only currently active clips load.
  // Cleanup AudioContext on unmount
  useEffect(() => {
    return () => {
      // Stop all sources
      for (const [itemId] of activeSourcesRef.current) {
        stopSource(itemId);
      }
      activeSourcesRef.current.clear();

      // Clean up audio graph
      if (masterGainRef.current) {
        masterGainRef.current.disconnect();
        masterGainRef.current = null;
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {
          // Ignore errors when closing
        });
        audioContextRef.current = null;
      }

      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [stopSource]);

  // Find ALL items that should be playing at a given story time
  const findActiveItems = useCallback(
    (storyTimeMs: number, itemList: StoryItemDetail[]): StoryItemDetail[] => {
      return itemList.filter((item) => {
        const itemStart = item.start_time_ms;
        // Use effective duration (accounting for trims)
        const trimStartMs = item.trim_start_ms || 0;
        const trimEndMs = item.trim_end_ms || 0;
        const effectiveDurationMs = item.duration * 1000 - trimStartMs - trimEndMs;
        const itemEnd = item.start_time_ms + effectiveDurationMs;
        return storyTimeMs >= itemStart && storyTimeMs < itemEnd;
      });
    },
    [],
  );

  // Convert AudioContext time to story time (ms)
  const contextTimeToStoryTime = useCallback(
    (contextTime: number): number => {
      if (playbackStartContextTime === null || playbackStartStoryTime === null) {
        return 0;
      }
      const elapsedContextTime = contextTime - playbackStartContextTime;
      return playbackStartStoryTime + elapsedContextTime * 1000;
    },
    [playbackStartContextTime, playbackStartStoryTime],
  );

  // Stop all sources
  const stopAllSources = useCallback(() => {
    console.log('[StoryPlayback] Stopping all sources');
    for (const [itemId] of activeSourcesRef.current) {
      stopSource(itemId);
    }
    activeSourcesRef.current.clear();
  }, [stopSource]);

  // Schedule playback for all items that should be playing
  const schedulePlayback = useCallback(
    (storyTimeMs: number, itemList: StoryItemDetail[]) => {
      const audioContext = getAudioContext();

      // Find all items that should be playing
      const shouldBePlaying = findActiveItems(storyTimeMs, itemList);
      if (shouldBePlaying.length > 32) {
        useStoryStore.getState().pause();
        toast({
          title: 'Too many overlapping clips',
          description: 'Play at most 32 simultaneous clips.',
          variant: 'destructive',
        });
        return;
      }

      const shouldBePlayingIds = new Set(shouldBePlaying.map((item) => item.id));

      // Stop sources that shouldn't be playing anymore
      for (const [itemId] of activeSourcesRef.current) {
        if (!shouldBePlayingIds.has(itemId)) {
          stopSource(itemId);
        }
      }

      // Schedule new sources for items that should be playing
      for (const item of shouldBePlaying) {
        if (!activeSourcesRef.current.has(item.id)) {
          // Calculate effective duration and trim offsets
          const trimStartSec = (item.trim_start_ms || 0) / 1000;
          const trimEndSec = (item.trim_end_ms || 0) / 1000;
          const effectiveDuration = item.duration - trimStartSec - trimEndSec;
          const itemEndStoryTime = item.start_time_ms + effectiveDuration * 1000;

          const audio = new Audio();
          audio.crossOrigin = 'anonymous';
          audio.preload = 'metadata';
          const source = audioContext.createMediaElementSource(audio);
          // Per-clip gain so each item can override its level independently
          // of the master volume. Falls through 1.0 for any item without a
          // saved value (older rows pre-migration).
          const clipGain = audioContext.createGain();
          clipGain.gain.value = typeof item.volume === 'number' ? item.volume : 1;
          source.connect(clipGain);
          clipGain.connect(masterGainRef.current || audioContext.destination);

          const activeSource: ActiveSource = {
            source,
            audio,
            request: new AbortController(),
            clipGain,
            itemId: item.id,
            generationId: item.generation_id,
            startTimeMs: item.start_time_ms,
            endTimeMs: itemEndStoryTime,
          };

          activeSourcesRef.current.set(item.id, activeSource);

          audio.onloadedmetadata = () => {
            if (activeSourcesRef.current.get(item.id) !== activeSource) return;
            const elapsed = (useStoryStore.getState().currentTimeMs - item.start_time_ms) / 1000;
            audio.currentTime = trimStartSec + Math.max(0, elapsed);
            audio.play().catch((error) => {
              if (activeSourcesRef.current.get(item.id) !== activeSource) return;
              useStoryStore.getState().pause();
              toast({
                title: 'Playback failed',
                description: String(error),
                variant: 'destructive',
              });
            });
          };
          audio.onerror = () => {
            if (activeSourcesRef.current.get(item.id) !== activeSource) return;
            useStoryStore.getState().pause();
            toast({ title: 'Could not load story audio', variant: 'destructive' });
          };
          audio.onended = () => stopSource(item.id);
          const url = item.version_id
            ? apiClient.getVersionAudioUrl(item.version_id)
            : apiClient.getAudioUrl(item.generation_id);
          void loadAudioSource(url, activeSource.request.signal)
            .then((src) => {
              if (activeSourcesRef.current.get(item.id) !== activeSource) {
                releaseAudioSource(src);
                return;
              }
              audio.src = src;
              audio.load();
            })
            .catch((error) => {
              if (activeSource.request.signal.aborted) return;
              useStoryStore.getState().pause();
              toast({
                title: 'Could not load story audio',
                description: String(error),
                variant: 'destructive',
              });
            });
        }
      }
    },
    [getAudioContext, findActiveItems, stopSource],
  );

  // Sync visual playhead from AudioContext time
  useEffect(() => {
    if (!isPlaying || playbackStartContextTime === null || playbackStartStoryTime === null) {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      return;
    }

    const audioContext = getAudioContext();
    const itemList = playbackItems || [];

    const syncPlayhead = () => {
      if (!useStoryStore.getState().isPlaying) {
        return;
      }

      const currentStoryTime = contextTimeToStoryTime(audioContext.currentTime);
      const totalDuration = useStoryStore.getState().totalDurationMs;

      // Update store with current story time
      useStoryStore.setState({ currentTimeMs: Math.min(currentStoryTime, totalDuration) });

      // Schedule any items that should be playing
      schedulePlayback(currentStoryTime, itemList);

      // Check if we've reached the end
      if (currentStoryTime >= totalDuration) {
        // Check if all sources have ended
        if (activeSourcesRef.current.size === 0) {
          console.log('[StoryPlayback] Reached end');
          useStoryStore.getState().stop();
          return;
        }
      }

      // Continue sync loop
      animationFrameRef.current = requestAnimationFrame(syncPlayhead);
    };

    // Initial sync
    const currentContextTime = audioContext.currentTime;
    const currentStoryTime = contextTimeToStoryTime(currentContextTime);
    schedulePlayback(currentStoryTime, itemList);

    // Start sync loop
    animationFrameRef.current = requestAnimationFrame(syncPlayhead);

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [
    isPlaying,
    playbackItems,
    playbackStartContextTime,
    playbackStartStoryTime,
    getAudioContext,
    contextTimeToStoryTime,
    schedulePlayback,
  ]);

  // Handle play/pause changes - stop sources when paused
  useEffect(() => {
    if (!isPlaying) {
      console.log('[StoryPlayback] Stopping playback');
      stopAllSources();
    }
  }, [isPlaying, stopAllSources]);

  // Handle seek - reset timing anchors when they become null (triggered by seek)
  useEffect(() => {
    if (!isPlaying || !playbackItems || playbackItems.length === 0) {
      return;
    }

    // Only run when timing anchors are null (after a seek)
    if (playbackStartContextTime !== null && playbackStartStoryTime !== null) {
      return;
    }

    const audioContext = getAudioContext();
    const currentContextTime = audioContext.currentTime;
    const currentStoryTime = useStoryStore.getState().currentTimeMs;

    console.log('[StoryPlayback] Setting timing anchors after seek:', {
      contextTime: currentContextTime,
      storyTime: currentStoryTime,
    });
    setPlaybackTiming(currentContextTime, currentStoryTime);

    // Stop all existing sources and reschedule from new position
    stopAllSources();
    schedulePlayback(currentStoryTime, playbackItems);
  }, [
    isPlaying,
    playbackItems,
    playbackStartContextTime,
    playbackStartStoryTime,
    getAudioContext,
    stopAllSources,
    schedulePlayback,
    setPlaybackTiming,
  ]);
}
