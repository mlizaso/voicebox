import { useEffect, useState } from 'react';
import { apiClient } from '@/lib/api/client';
import type { ActiveDownloadTask } from '@/lib/api/types';
import { useGenerationStore } from '@/stores/generationStore';
import { useServerStore } from '@/stores/serverStore';

// Polling interval in milliseconds
const POLL_INTERVAL = 30000;

/**
 * Hook to monitor active tasks (downloads and generations).
 * Polls the server periodically to catch downloads triggered from anywhere
 * (transcription, generation, explicit download, etc.).
 *
 * Returns the active downloads so components can render download toasts.
 */
export function useRestoreActiveTasks() {
  const connectionId = useServerStore((state) => state.connectionId);
  const [activeDownloads, setActiveDownloads] = useState<{
    connectionId: string;
    downloads: ActiveDownloadTask[];
  }>({ connectionId, downloads: [] });
  const setActiveGenerationId = useGenerationStore((state) => state.setActiveGenerationId);
  const addPendingGeneration = useGenerationStore((state) => state.addPendingGeneration);

  useEffect(() => {
    let active = true;
    let fetching = false;
    const fetchActiveTasks = async () => {
      if (!active || fetching || connectionId !== useServerStore.getState().connectionId) return;
      fetching = true;
      try {
        const tasks = await apiClient.getActiveTasks();
        if (!active || connectionId !== useServerStore.getState().connectionId) return;

        // Restore pending generations (e.g., after page refresh).
        if (tasks.generations.length > 0) {
          setActiveGenerationId(tasks.generations[0].task_id);
          for (const gen of tasks.generations) {
            addPendingGeneration(gen.task_id);
          }
        } else {
          setActiveGenerationId(null);
        }

        setActiveDownloads({ connectionId, downloads: tasks.downloads });
      } catch (error) {
        if (active && connectionId === useServerStore.getState().connectionId) {
          // The server might be temporarily unavailable.
          console.debug('Failed to fetch active tasks:', error);
        }
      } finally {
        fetching = false;
      }
    };

    // Fetch immediately for each connection and avoid overlapping polls.
    fetchActiveTasks();
    const interval = setInterval(fetchActiveTasks, POLL_INTERVAL);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [connectionId, setActiveGenerationId, addPendingGeneration]);

  return activeDownloads.connectionId === connectionId ? activeDownloads.downloads : [];
}

/**
 * Map model names to display names for download toasts.
 */
export const MODEL_DISPLAY_NAMES: Record<string, string> = {
  'qwen-tts-1.7B': 'Qwen TTS 1.7B',
  'qwen-tts-0.6B': 'Qwen TTS 0.6B',
  'whisper-base': 'Whisper Base',
  'whisper-small': 'Whisper Small',
  'whisper-medium': 'Whisper Medium',
  'whisper-large': 'Whisper Large',
};
