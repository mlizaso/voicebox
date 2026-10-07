import { useServerStore } from '@/stores/serverStore';

/** A multi-step operation belongs to the connection on which it began,
 * including when local decoding leaves no pending request/mutation. */
export function captureServerConnection(message: string) {
  const connectionId = useServerStore.getState().connectionId;
  const isCurrent = () => useServerStore.getState().connectionId === connectionId;
  return {
    isCurrent,
    assertCurrent() {
      if (!isCurrent()) throw new Error(message);
    },
  };
}
