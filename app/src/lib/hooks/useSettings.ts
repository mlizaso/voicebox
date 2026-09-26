import { apiClient } from '@/lib/api/client';
import type { CaptureSettings, GenerationSettings } from '@/lib/api/types';
import { useSettingsResource } from './useSettingsResource';

const CAPTURE_SETTINGS_KEY = ['settings', 'captures'] as const;
const GENERATION_SETTINGS_KEY = ['settings', 'generation'] as const;

export function useCaptureSettings() {
  return useSettingsResource<CaptureSettings>(
    CAPTURE_SETTINGS_KEY,
    () => apiClient.getCaptureSettings(),
    (patch) => apiClient.updateCaptureSettings(patch),
    (queryClient, patch) => {
      // Model swaps change readiness immediately; do not wait for its poll.
      if (patch.stt_model !== undefined || patch.llm_model !== undefined) {
        void queryClient.invalidateQueries({ queryKey: ['capture-readiness'] });
      }
    },
  );
}

export function useGenerationSettings() {
  return useSettingsResource<GenerationSettings>(
    GENERATION_SETTINGS_KEY,
    () => apiClient.getGenerationSettings(),
    (patch) => apiClient.updateGenerationSettings(patch),
  );
}
