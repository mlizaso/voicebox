import { invoke } from '@tauri-apps/api/core';
import { emit } from '@tauri-apps/api/event';
import type {
  PlatformLifecycle,
  ServerCloseState,
  ServerConnection,
  ServerLogEntry,
} from '@/platform/types';
import { subscribeToTauriEvent } from './listener';

class TauriLifecycle implements PlatformLifecycle {
  onServerReady?: () => void;
  onServerStopped?: () => void;
  private stoppedEpoch = 0;

  async setClientConnection(connection: ServerConnection): Promise<void> {
    await invoke('set_client_connection', { connection });
  }

  async getClientConnection(): Promise<ServerConnection | null> {
    return invoke<ServerConnection | null>('get_client_connection');
  }

  async startServer(
    remote = false,
    modelsDir?: string | null,
    remoteApiToken?: string | null,
    allowExternal = false,
  ): Promise<string> {
    const epoch = this.stoppedEpoch;
    try {
      const result = await invoke<string>('start_server', {
        remote,
        allowExternal,
        modelsDir: modelsDir ?? undefined,
        remoteApiToken: remoteApiToken ?? undefined,
      });
      if (epoch !== this.stoppedEpoch) throw new Error('Server stopped during startup');
      console.log('Server started:', result);
      this.onServerReady?.();
      return result;
    } catch (error) {
      console.error('Failed to start server:', error);
      throw error;
    }
  }

  async stopServer(): Promise<void> {
    try {
      await invoke('stop_server');
      this.onServerStopped?.();
      console.log('Server stopped');
    } catch (error) {
      console.error('Failed to stop server:', error);
      throw error;
    }
  }

  async restartServer(modelsDir?: string | null): Promise<string> {
    this.onServerStopped?.();
    const epoch = this.stoppedEpoch;
    try {
      const result = await invoke<string>('restart_server', {
        modelsDir: modelsDir ?? undefined,
      });
      if (epoch !== this.stoppedEpoch) throw new Error('Server stopped during restart');
      console.log('Server restarted:', result);
      this.onServerReady?.();
      return result;
    } catch (error) {
      console.error('Failed to restart server:', error);
      throw error;
    }
  }

  async setKeepServerRunning(keepRunning: boolean): Promise<void> {
    try {
      await invoke('set_keep_server_running', { keepRunning });
    } catch (error) {
      console.error('Failed to set keep server running setting:', error);
    }
  }

  async setBackendOverride(backend?: string | null): Promise<void> {
    try {
      await invoke('set_backend_override', { backend: backend ?? undefined });
    } catch (error) {
      console.error('Failed to set backend override:', error);
      throw error;
    }
  }

  subscribeToWindowClose(getState: () => ServerCloseState): () => void {
    return subscribeToTauriEvent<null>('window-close-requested', async () => {
      const { keepServerRunning, serverStartedByApp } = getState();

      console.log(
        '[lifecycle] window-close-requested: keepRunning=%s, serverStartedByApp=%s',
        keepServerRunning,
        serverStartedByApp,
      );

      if (!keepServerRunning && serverStartedByApp) {
        try {
          await this.stopServer();
        } catch (error) {
          console.error('Failed to stop server on close:', error);
        }
      }

      await emit('window-close-allowed');
    });
  }

  subscribeToServerLogs(callback: (entry: ServerLogEntry) => void): () => void {
    const logs = subscribeToTauriEvent('server-log', callback);
    const stopped = subscribeToTauriEvent('server-stopped', () => {
      this.stoppedEpoch += 1;
      this.onServerStopped?.();
    });
    return () => {
      logs();
      stopped();
    };
  }
}

export const tauriLifecycle = new TauriLifecycle();
