import type { PlatformDictation } from '@/platform/types';

/** Native dictation integration is intentionally unavailable in browsers. */
export const webDictation: PlatformDictation = {
  setHotkeysEnabled() {
    return Promise.resolve();
  },
  checkAccessibilityPermission() {
    return Promise.resolve(true);
  },
  openAccessibilitySettings() {
    return Promise.resolve();
  },
  checkInputMonitoringPermission() {
    return Promise.resolve(true);
  },
  openInputMonitoringSettings() {
    return Promise.resolve();
  },
  pasteFinalText() {
    return Promise.resolve(false);
  },
};
