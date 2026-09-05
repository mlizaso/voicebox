import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { toast } from '@/components/ui/use-toast';
import { queryClient } from '@/lib/queryClient';
import type { ServerConnection } from '@/platform/types';
import { useGenerationStore } from './generationStore';
import { usePlayerStore } from './playerStore';
import { useStoryStore } from './storyStore';
import { useUIStore } from './uiStore';

interface ServerStore extends ServerConnection {
  setConnection: (connection: ServerConnection) => boolean;
  setServerUrl: (url: string) => void;

  setRemoteApiToken: (token: string) => void;

  isConnected: boolean;
  setIsConnected: (connected: boolean) => void;

  setMode: (mode: 'local' | 'remote') => void;

  keepServerRunningOnClose: boolean;
  setKeepServerRunningOnClose: (keepRunning: boolean) => void;

  customModelsDir: string | null;
  setCustomModelsDir: (dir: string | null) => void;
}

function canChangeConnection(): boolean {
  if (queryClient.isMutating() === 0) return true;
  toast({
    title: 'An operation is still running',
    description: 'Wait for it to finish before changing servers or credentials.',
    variant: 'destructive',
  });
  return false;
}

function newConnectionId(): string {
  // getRandomValues also works when the web UI was opened over plain HTTP;
  // the connection validator can then explain why remote requests are blocked.
  return crypto.getRandomValues(new Uint32Array(4)).join('-');
}

/** Drop the previous server's data before requesting the new identity's data. */
function resetServerData() {
  usePlayerStore.getState().reset();
  useStoryStore.getState().stop();
  useStoryStore.setState({ selectedStoryId: null, selectedClipId: null });
  useGenerationStore.setState({
    pendingGenerationIds: new Set(),
    pendingStoryAdds: new Map(),
    isGenerating: false,
    activeGenerationId: null,
  });
  useUIStore.setState({
    selectedProfileId: null,
    selectedVoiceId: null,
    editingProfileId: null,
    profileFormDraft: null,
    profileDialogOpen: false,
    generationDialogOpen: false,
  });
  queryClient.getMutationCache().clear();
  // resetQueries cancels in-flight queries, clears their data, notifies mounted
  // observers and refetches active queries using the updated connection.
  void queryClient.resetQueries();
}

export function getDefaultServerUrl(): string {
  const fallback = 'http://127.0.0.1:17493';

  if (!import.meta.env.PROD || typeof window === 'undefined') {
    return fallback;
  }

  const { protocol, origin, hostname } = window.location;
  if ((protocol === 'http:' || protocol === 'https:') && origin && hostname !== 'tauri.localhost') {
    return origin;
  }

  return fallback;
}

export function isLoopbackVoiceboxServerUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return (
      parsed.port === '17493' &&
      (parsed.hostname === '127.0.0.1' ||
        parsed.hostname === 'localhost' ||
        parsed.hostname === '[::1]' ||
        parsed.hostname === '::1')
    );
  } catch {
    return false;
  }
}

export const useServerStore = create<ServerStore>()(
  persist(
    (set, get) => ({
      connectionId: newConnectionId(),
      setConnection: (connection) => {
        if (connection.connectionId === get().connectionId) return true;
        if (!canChangeConnection()) return false;
        set({ ...connection, isConnected: false });
        resetServerData();
        return true;
      },
      serverUrl: getDefaultServerUrl(),
      setServerUrl: (url) => {
        const prev = get().serverUrl;
        if (url !== prev && canChangeConnection()) {
          set({
            serverUrl: url,
            remoteApiToken: '',
            connectionId: newConnectionId(),
            isConnected: false,
          });
          resetServerData();
        }
      },

      remoteApiToken: '',
      setRemoteApiToken: (token) => {
        const prev = get().remoteApiToken;
        if (token !== prev && canChangeConnection()) {
          set({ remoteApiToken: token, connectionId: newConnectionId(), isConnected: false });
          resetServerData();
        }
      },

      isConnected: false,
      setIsConnected: (connected) => set({ isConnected: connected }),

      mode: 'local',
      setMode: (mode) => {
        if (mode !== get().mode && canChangeConnection()) {
          set({ mode, connectionId: newConnectionId(), isConnected: false });
          resetServerData();
        }
      },

      keepServerRunningOnClose: false,
      setKeepServerRunningOnClose: (keepRunning) => set({ keepServerRunningOnClose: keepRunning }),

      customModelsDir: null,
      setCustomModelsDir: (dir) => set({ customModelsDir: dir }),
    }),
    {
      name: 'voicebox-server',
      version: 1,
      partialize: (state) => ({
        serverUrl: state.serverUrl,
        mode: state.mode,
        keepServerRunningOnClose: state.keepServerRunningOnClose,
        customModelsDir: state.customModelsDir,
      }),
      // Persist only connection preferences. Upgrading rewrites storage without
      // the reusable bearer token previously saved by version 0.
      migrate: (persisted) => {
        const {
          remoteApiToken: _token,
          isConnected: _connected,
          ...preferences
        } = persisted as ServerStore;
        return preferences;
      },
    },
  ),
);
