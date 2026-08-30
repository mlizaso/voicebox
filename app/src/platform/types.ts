import type { CaptureResponse } from '@/lib/api/types';

/**
 * Platform abstraction types
 * These interfaces define the contract that platform implementations must fulfill
 */

export interface FileFilter {
  name: string;
  extensions: string[];
}

export interface SavedFile {
  displayName: string;
  outcome: 'saved' | 'download-started';
}

export interface PlatformFilesystem {
  saveFile(filename: string, blob: Blob, filters?: FileFilter[]): Promise<SavedFile | null>;
  saveResponse(
    filename: string,
    getResponse: () => Promise<Response>,
    maxBytes: number,
    filters?: FileFilter[],
  ): Promise<SavedFile | null>;
  openPath(path: string): Promise<void>;
  pickDirectory(title: string): Promise<string | null>;
}

export interface FocusSnapshot {
  pid: number;
  bundle_id: string | null;
  role: string | null;
}

/** Every cross-window event supported by the shared application layer. */
export interface PlatformEventMap {
  'capture:created': { capture: CaptureResponse };
  'capture:updated': { id: string };
  'system:accessibility-missing': undefined;
  'dictate:start': { focus: FocusSnapshot | null };
  'dictate:stop': undefined;
  'dictate:restart': undefined;
  'dictate:speak-start': string;
  'dictate:speak-end': string;
  'dictate:show': undefined;
  'dictate:hide': undefined;
}

export type PlatformEventPayload<K extends keyof PlatformEventMap> =
  PlatformEventMap[K] extends undefined ? [] : [payload: PlatformEventMap[K]];

export interface PlatformEvents {
  subscribe<K extends keyof PlatformEventMap>(
    eventName: K,
    handler: (payload: PlatformEventMap[K]) => void,
  ): () => void;
  emit<K extends keyof PlatformEventMap>(
    eventName: K,
    ...payload: PlatformEventPayload<K>
  ): Promise<void>;
}

export interface HotkeyBindings {
  pushToTalk: string[];
  toggleToTalk: string[];
}

export interface PlatformDictation {
  setHotkeysEnabled(enabled: boolean, bindings: HotkeyBindings): Promise<void>;
  checkAccessibilityPermission(): Promise<boolean>;
  openAccessibilitySettings(): Promise<void>;
  checkInputMonitoringPermission(): Promise<boolean>;
  openInputMonitoringSettings(): Promise<void>;
  pasteFinalText(text: string, focus: FocusSnapshot): Promise<boolean>;
}

export interface UpdateStatus {
  checking: boolean;
  available: boolean;
  version?: string;
  downloading: boolean;
  installing: boolean;
  readyToInstall: boolean;
  error?: string;
  downloadProgress?: number; // 0-100 percentage
  downloadedBytes?: number;
  totalBytes?: number;
}

export interface PlatformUpdater {
  checkForUpdates(): Promise<void>;
  downloadAndInstall(): Promise<void>;
  restartAndInstall(): Promise<void>;
  getStatus(): UpdateStatus;
  subscribe(callback: (status: UpdateStatus) => void): () => void;
}

export interface AudioDevice {
  id: string;
  name: string;
  is_default: boolean;
}

export interface PlatformAudio {
  isSystemAudioSupported(): Promise<boolean>;
  startSystemAudioCapture(maxDurationSecs: number): Promise<void>;
  stopSystemAudioCapture(): Promise<Blob>;
  listOutputDevices(): Promise<AudioDevice[]>;
  playToDevices(audioData: Uint8Array, deviceIds: string[]): Promise<void>;
  stopPlayback(): void;
}

export interface ServerLogEntry {
  stream: 'stdout' | 'stderr';
  line: string;
}

export interface ServerCloseState {
  keepServerRunning: boolean;
  serverStartedByApp: boolean;
}

export interface PlatformLifecycle {
  startServer(
    remote?: boolean,
    modelsDir?: string | null,
    remoteApiToken?: string | null,
  ): Promise<string>;
  stopServer(): Promise<void>;
  restartServer(modelsDir?: string | null): Promise<string>;
  setKeepServerRunning(keep: boolean): Promise<void>;
  setBackendOverride(backend?: string | null): Promise<void>;
  subscribeToWindowClose(getState: () => ServerCloseState): () => void;
  subscribeToServerLogs(callback: (entry: ServerLogEntry) => void): () => void;
  onServerReady?: () => void;
}

export interface PlatformMetadata {
  getVersion(): Promise<string>;
  isTauri: boolean;
}

export interface Platform {
  filesystem: PlatformFilesystem;
  events: PlatformEvents;
  dictation: PlatformDictation;
  updater: PlatformUpdater;
  audio: PlatformAudio;
  lifecycle: PlatformLifecycle;
  metadata: PlatformMetadata;
}
