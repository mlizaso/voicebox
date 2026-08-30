import { emit as emitTauriEvent } from '@tauri-apps/api/event';
import type { PlatformEventMap, PlatformEventPayload, PlatformEvents } from '@/platform/types';
import { subscribeToTauriEvent } from './listener';

export const tauriEvents: PlatformEvents = {
  subscribe<K extends keyof PlatformEventMap>(
    eventName: K,
    handler: (payload: PlatformEventMap[K]) => void,
  ) {
    return subscribeToTauriEvent(eventName, handler);
  },

  emit<K extends keyof PlatformEventMap>(eventName: K, ...payload: PlatformEventPayload<K>) {
    return emitTauriEvent(eventName, payload[0]);
  },
};
