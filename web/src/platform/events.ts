import type { PlatformEventMap, PlatformEventPayload, PlatformEvents } from '@/platform/types';

/** Browser deployments have no sibling webviews or native event bridge. */
export const webEvents: PlatformEvents = {
  subscribe<K extends keyof PlatformEventMap>(
    _eventName: K,
    _handler: (payload: PlatformEventMap[K]) => void,
  ) {
    return () => {};
  },

  emit<K extends keyof PlatformEventMap>(_eventName: K, ..._payload: PlatformEventPayload<K>) {
    return Promise.resolve();
  },
};
