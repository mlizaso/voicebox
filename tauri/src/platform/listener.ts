import { listen } from '@tauri-apps/api/event';

type TauriUnlisten = () => void | Promise<void>;

function reportListenerError(eventName: string, action: string, error: unknown) {
  console.warn(`[platform-events] failed to ${action} ${eventName}:`, error);
}

function safelyUnlisten(eventName: string, unlisten: TauriUnlisten) {
  try {
    void Promise.resolve(unlisten()).catch((error) => {
      reportListenerError(eventName, 'unsubscribe from', error);
    });
  } catch (error) {
    reportListenerError(eventName, 'unsubscribe from', error);
  }
}

/** Register a Tauri listener with synchronous, StrictMode-safe disposal. */
export function subscribeToTauriEvent<T>(
  eventName: string,
  handler: (payload: T) => void | Promise<void>,
): () => void {
  let disposed = false;
  let unlisten: TauriUnlisten | null = null;

  void listen<T>(eventName, (event) => {
    if (disposed) return;
    try {
      void Promise.resolve(handler(event.payload)).catch((error) => {
        reportListenerError(eventName, 'handle', error);
      });
    } catch (error) {
      reportListenerError(eventName, 'handle', error);
    }
  })
    .then((registeredUnlisten) => {
      if (disposed) {
        safelyUnlisten(eventName, registeredUnlisten);
        return;
      }
      unlisten = registeredUnlisten;
    })
    .catch((error) => {
      reportListenerError(eventName, 'subscribe to', error);
    });

  return () => {
    disposed = true;
    if (unlisten) safelyUnlisten(eventName, unlisten);
    unlisten = null;
  };
}
