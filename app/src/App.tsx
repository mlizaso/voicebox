import { RouterProvider } from '@tanstack/react-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import voiceboxLogo from '@/assets/voicebox-logo.png';
import { DictateWindow } from '@/components/DictateWindow/DictateWindow';
import ShinyText from '@/components/ShinyText';
import { TitleBarDragRegion } from '@/components/TitleBarDragRegion';
import { Input } from '@/components/ui/input';
import { useAutoUpdater } from '@/hooks/useAutoUpdater';
import { useThemeSync } from '@/hooks/useThemeSync';
import { TOP_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import { useChordSync } from '@/lib/hooks/useChordSync';
import { cn } from '@/lib/utils/cn';
import { usePlatform } from '@/platform/PlatformContext';
import { router } from '@/router';
import { useLogStore } from '@/stores/logStore';
import {
  getDefaultServerUrl,
  isLoopbackVoiceboxServerUrl,
  useServerStore,
} from '@/stores/serverStore';

function isDictateView(): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('view') === 'dictate';
}

const LOADING_MESSAGES = [
  'Warming up tensors...',
  'Calibrating synthesizer engine...',
  'Initializing voice models...',
  'Loading neural networks...',
  'Preparing audio pipelines...',
  'Optimizing waveform generators...',
  'Tuning frequency analyzers...',
  'Building voice embeddings...',
  'Configuring text-to-speech cores...',
  'Syncing audio buffers...',
  'Establishing model connections...',
  'Preprocessing training data...',
  'Validating voice samples...',
  'Compiling inference engines...',
  'Mapping phoneme sequences...',
  'Aligning prosody parameters...',
  'Activating speech synthesis...',
  'Fine-tuning acoustic models...',
  'Preparing voice cloning matrices...',
  'Initializing Qwen TTS framework...',
];

function App() {
  useThemeSync();
  const platform = usePlatform();

  // The dictate window runs in a separate Tauri webview that must skip
  // server bootstrap (the main window owns that lifecycle) and render only
  // the floating recording surface. Split into a sibling component so the
  // main app's hooks are not called on the dictate path.
  if (platform.metadata.isTauri && isDictateView()) {
    return <DictateWindow />;
  }
  return <MainApp />;
}

function MainApp() {
  const platform = usePlatform();
  const [serverReady, setServerReady] = useState(false);
  const [startupError, setStartupError] = useState<string | null>(null);
  const [loadingMessageIndex, setLoadingMessageIndex] = useState(0);
  const serverStartingRef = useRef(false);
  const remoteApiToken = useServerStore((state) => state.remoteApiToken);

  const startServer = useCallback(
    async (allowExternal = false) => {
      if (serverStartingRef.current) return;
      serverStartingRef.current = true;
      setStartupError(null);
      const state = useServerStore.getState();
      try {
        const url = await platform.lifecycle.startServer(
          state.mode === 'remote',
          state.customModelsDir,
          state.remoteApiToken || null,
          allowExternal,
        );
        state.setServerUrl(url);
        window.__voiceboxServerStartedByApp = !allowExternal;
        setServerReady(true);
      } catch (error) {
        window.__voiceboxServerStartedByApp = false;
        setStartupError(error instanceof Error ? error.message : String(error));
      } finally {
        serverStartingRef.current = false;
      }
    },
    [platform.lifecycle],
  );

  useEffect(() => {
    if (!platform.metadata.isTauri) return;
    const syncConnection = () => {
      const { connectionId, serverUrl, remoteApiToken, mode } = useServerStore.getState();
      void platform.lifecycle
        .setClientConnection({ connectionId, serverUrl, remoteApiToken, mode })
        .catch(() => console.error('Could not share the connection with the dictation window'));
    };
    syncConnection();
    return useServerStore.subscribe((state, previous) => {
      if (state.connectionId !== previous.connectionId) syncConnection();
    });
  }, [platform.lifecycle, platform.metadata.isTauri]);

  // Automatically check for app updates on startup and show toast notifications
  useAutoUpdater({ checkOnMount: true, showToast: true });

  // Sync stored setting to Rust on startup
  useEffect(() => {
    if (platform.metadata.isTauri) {
      const keepRunning = useServerStore.getState().keepServerRunningOnClose;
      platform.lifecycle.setKeepServerRunning(keepRunning).catch((error) => {
        console.error('Failed to sync initial setting to Rust:', error);
      });
    }
    // Empty dependency array - platform is stable from context, only run once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform.metadata.isTauri, platform.lifecycle]);

  // Setup lifecycle callbacks
  useEffect(() => {
    platform.lifecycle.onServerReady = () => {
      setServerReady(true);
      setStartupError(null);
    };
    platform.lifecycle.onServerStopped = () => {
      window.__voiceboxServerStartedByApp = false;
      if (!isLoopbackVoiceboxServerUrl(useServerStore.getState().serverUrl)) return;
      setServerReady(false);
      setStartupError('The server stopped. Retry to reconnect.');
    };
    return () => {
      platform.lifecycle.onServerReady = undefined;
      platform.lifecycle.onServerStopped = undefined;
    };
    // Empty dependency array - platform is stable from context, only run once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform.lifecycle]);

  // Subscribe to server logs
  useEffect(() => {
    const unsubscribe = platform.lifecycle.subscribeToServerLogs((entry) => {
      useLogStore.getState().addEntry(entry);
    });
    return unsubscribe;
  }, [platform.lifecycle]);

  useEffect(() => {
    if (!platform.metadata.isTauri) return;
    return platform.lifecycle.subscribeToWindowClose(() => ({
      keepServerRunning: useServerStore.getState().keepServerRunningOnClose,
      serverStartedByApp: window.__voiceboxServerStartedByApp ?? false,
    }));
  }, [platform.lifecycle, platform.metadata.isTauri]);

  // Auto-start the bundled server when running in Tauri production mode.
  useEffect(() => {
    if (!platform.metadata.isTauri) {
      const serverUrl = getDefaultServerUrl();
      const currentServerUrl = useServerStore.getState().serverUrl;
      if (currentServerUrl !== serverUrl && isLoopbackVoiceboxServerUrl(currentServerUrl)) {
        useServerStore.getState().setServerUrl(serverUrl);
      }
      setServerReady(true); // Web assumes server is running
      return;
    }

    // An explicitly configured external server does not need a local sidecar.
    // Show settings so the user can enter its memory-only token each session.
    if (!isLoopbackVoiceboxServerUrl(useServerStore.getState().serverUrl)) {
      window.__voiceboxServerStartedByApp = false;
      setServerReady(true);
      return;
    }

    // Only auto-start server in production mode
    // In dev mode, user runs server separately
    if (!import.meta.env?.PROD) {
      console.log('Dev mode: Skipping auto-start of server (run it separately)');
      setServerReady(true); // Mark as ready so UI doesn't show loading screen
      // Mark that server was not started by app (so we don't try to stop it on close)
      window.__voiceboxServerStartedByApp = false;
      return;
    }

    void startServer();

    // Empty dependency array - platform is stable from context, only run once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform.metadata.isTauri, startServer]);

  // Cycle through loading messages every 3 seconds
  useEffect(() => {
    if (!platform.metadata.isTauri || serverReady) {
      return;
    }

    const interval = setInterval(() => {
      setLoadingMessageIndex((prev) => (prev + 1) % LOADING_MESSAGES.length);
    }, 3000);

    return () => clearInterval(interval);
  }, [serverReady, platform.metadata.isTauri]);

  // Show loading screen while server is starting in Tauri
  if (platform.metadata.isTauri && !serverReady) {
    return (
      <div
        className={cn(
          'min-h-screen bg-background flex items-center justify-center',
          TOP_SAFE_AREA_PADDING,
        )}
      >
        <TitleBarDragRegion />
        <div className="text-center space-y-6">
          <div className="flex justify-center relative">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-48 h-48 rounded-full bg-accent/20 blur-3xl" />
            </div>
            <img
              src={voiceboxLogo}
              alt="Voicebox"
              className="w-48 h-48 object-contain animate-fade-in-scale relative z-10"
            />
          </div>
          {startupError ? (
            <div className="animate-fade-in-delayed max-w-md mx-auto space-y-3">
              <p className="text-lg font-medium text-destructive">Server startup failed</p>
              <p className="text-sm text-muted-foreground">{startupError}</p>
              {useServerStore.getState().mode === 'remote' && (
                <Input
                  type="password"
                  autoComplete="off"
                  spellCheck={false}
                  aria-label="Remote API token for this session"
                  placeholder="Remote API token for this session"
                  value={remoteApiToken}
                  onChange={(event) =>
                    useServerStore.getState().setRemoteApiToken(event.target.value)
                  }
                />
              )}
              {startupError.includes('already in use') && (
                <button
                  type="button"
                  className="px-4 py-2 text-sm rounded-md border hover:bg-accent"
                  onClick={() => void startServer(true)}
                >
                  Connect to my running server
                </button>
              )}
              <button
                type="button"
                className="mt-2 px-4 py-2 text-sm rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                onClick={() => void startServer()}
              >
                Retry
              </button>
            </div>
          ) : (
            <div className="animate-fade-in-delayed">
              <ShinyText
                text={LOADING_MESSAGES[loadingMessageIndex]}
                className="text-lg font-medium text-muted-foreground"
                speed={2}
                color="hsl(var(--muted-foreground))"
                shineColor="hsl(var(--foreground))"
              />
            </div>
          )}
        </div>
      </div>
    );
  }

  return <ConnectedApp />;
}

function ConnectedApp() {
  useChordSync();
  return <RouterProvider router={router} />;
}

export default App;
