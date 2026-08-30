import { invoke } from '@tauri-apps/api/core';
import type { FocusSnapshot, HotkeyBindings, PlatformDictation } from '@/platform/types';

export const tauriDictation: PlatformDictation = {
  setHotkeysEnabled(enabled: boolean, bindings: HotkeyBindings) {
    if (!enabled) {
      return invoke<void>('disable_hotkey');
    }
    return invoke<void>('enable_hotkey', {
      pushToTalk: bindings.pushToTalk,
      toggleToTalk: bindings.toggleToTalk,
    });
  },

  checkAccessibilityPermission() {
    return invoke<boolean>('check_accessibility_permission');
  },

  openAccessibilitySettings() {
    return invoke<void>('open_accessibility_settings');
  },

  checkInputMonitoringPermission() {
    return invoke<boolean>('check_input_monitoring_permission');
  },

  openInputMonitoringSettings() {
    return invoke<void>('open_input_monitoring_settings');
  },

  pasteFinalText(text: string, focus: FocusSnapshot) {
    return invoke<boolean>('paste_final_text', { text, focus });
  },
};
