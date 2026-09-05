import type {
  PlatformLifecycle,
  ServerCloseState,
  ServerConnection,
  ServerLogEntry,
} from '@/platform/types';
import { getDefaultServerUrl } from '@/stores/serverStore';

class WebLifecycle implements PlatformLifecycle {
  onServerReady?: () => void;

  async setClientConnection(_connection: ServerConnection): Promise<void> {}

  async getClientConnection(): Promise<ServerConnection | null> {
    return null; // Web has no separate dictation window.
  }

  async startServer(
    _remote = false,
    _modelsDir?: string | null,
    _remoteApiToken?: string | null,
  ): Promise<string> {
    // Web assumes server is running externally
    const serverUrl = import.meta.env.VITE_SERVER_URL || getDefaultServerUrl();
    this.onServerReady?.();
    return serverUrl;
  }

  async stopServer(): Promise<void> {
    // No-op for web - server is managed externally
  }

  async restartServer(_modelsDir?: string | null): Promise<string> {
    // No-op for web - server is managed externally
    return import.meta.env.VITE_SERVER_URL || getDefaultServerUrl();
  }

  async setKeepServerRunning(_keep: boolean): Promise<void> {
    // No-op for web
  }

  async setBackendOverride(_backend?: string | null): Promise<void> {
    // No-op for web - backend variant is managed externally
  }

  subscribeToWindowClose(_getState: () => ServerCloseState): () => void {
    return () => {};
  }

  subscribeToServerLogs(_callback: (_entry: ServerLogEntry) => void): () => void {
    // No-op for web - server logs are not available
    return () => {};
  }
}

export const webLifecycle = new WebLifecycle();
