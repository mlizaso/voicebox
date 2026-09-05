import { afterEach, expect, mock, test } from 'bun:test';
import { QueryClientProvider } from '@tanstack/react-query';
import { createElement } from 'react';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import type { StoryItemDetail } from '../src/lib/api/types';
import { useAudioRecording } from '../src/lib/hooks/useAudioRecording';
import { useCaptureRecordingSession } from '../src/lib/hooks/useCaptureRecordingSession';
import { useStoryPlayback } from '../src/lib/hooks/useStoryPlayback';
import { queryClient } from '../src/lib/queryClient';
import { PlatformProvider } from '../src/platform/PlatformContext';
import type { Platform } from '../src/platform/types';
import { useServerStore } from '../src/stores/serverStore';
import { useStoryStore } from '../src/stores/storyStore';

let renderer: ReactTestRenderer | undefined;
const originalWindow = globalThis.window;
const originalNavigator = globalThis.navigator;
const originalAudio = globalThis.Audio;
const originalContext = globalThis.AudioContext;
const originalFrame = globalThis.requestAnimationFrame;
const originalCancelFrame = globalThis.cancelAnimationFrame;
const originalMediaRecorder = globalThis.MediaRecorder;

afterEach(() => {
  act(() => renderer?.unmount());
  renderer = undefined;
  useStoryStore.getState().stop();
  Object.assign(globalThis, {
    window: originalWindow,
    Audio: originalAudio,
    AudioContext: originalContext,
    requestAnimationFrame: originalFrame,
    cancelAnimationFrame: originalCancelFrame,
    MediaRecorder: originalMediaRecorder,
  });
  Object.defineProperty(globalThis, 'navigator', { value: originalNavigator, configurable: true });
});

test.each([
  'unmount',
  'stop',
])('mount does not open a microphone and late permission is released after %s', async (end) => {
  const createRecorder = mock(() => {});
  Object.assign(globalThis, {
    MediaRecorder: class {
      static isTypeSupported() {
        return true;
      }
      constructor() {
        createRecorder();
      }
      start() {}
    },
  });
  let resolvePermission: (stream: MediaStream) => void = () => {};
  const getUserMedia = mock(
    () =>
      new Promise<MediaStream>((resolve) => {
        resolvePermission = resolve;
      }),
  );
  Object.defineProperty(globalThis, 'navigator', {
    value: { mediaDevices: { getUserMedia } },
    configurable: true,
  });
  let recording!: ReturnType<typeof useAudioRecording>;
  function Recorder() {
    recording = useAudioRecording();
    return null;
  }
  const platform = { metadata: { isTauri: false } } as Platform;
  act(() => {
    renderer = create(
      <PlatformProvider platform={platform}>
        <Recorder />
      </PlatformProvider>,
    );
  });
  expect(getUserMedia).not.toHaveBeenCalled();
  const pending = recording.startRecording();
  await recording.startRecording();
  expect(getUserMedia).toHaveBeenCalledTimes(1);
  act(() => {
    if (end === 'unmount') {
      renderer?.unmount();
      renderer = undefined;
    } else {
      recording.stopRecording();
    }
  });
  const stop = mock(() => {});
  resolvePermission({ getTracks: () => [{ stop }] } as unknown as MediaStream);
  await pending;
  expect(createRecorder).not.toHaveBeenCalled();
  expect(stop).toHaveBeenCalledTimes(1);
});

test('a recording cannot upload to a connection selected after it began', async () => {
  const originalRecorder = globalThis.MediaRecorder;
  const originalFetch = globalThis.fetch;
  const originalDateNow = Date.now;
  let now = 1000;
  Date.now = () => now;
  queryClient.clear();
  useServerStore.setState({
    connectionId: 'recording-server',
    serverUrl: 'http://127.0.0.1:17493',
    remoteApiToken: '',
  });
  let recorder!: FakeRecorder;
  class FakeRecorder {
    static isTypeSupported() {
      return true;
    }
    state = 'inactive';
    mimeType = 'audio/webm';
    onstop: (() => Promise<void>) | null = null;
    ondataavailable: ((event: { data: Blob }) => void) | null = null;
    constructor() {
      recorder = this;
    }
    start() {
      this.state = 'recording';
    }
    stop() {
      this.state = 'inactive';
    }
  }
  const trackStop = mock(() => {});
  const getUserMedia = mock(async () => ({ getTracks: () => [{ stop: trackStop }] }));
  Object.defineProperty(globalThis, 'navigator', {
    value: { mediaDevices: { getUserMedia } },
    configurable: true,
  });
  Object.assign(globalThis, {
    window: { setInterval, clearInterval, setTimeout, clearTimeout },
    MediaRecorder: FakeRecorder,
    AudioContext: class {
      async decodeAudioData() {
        throw new Error('Synthetic unsupported codec');
      }
    },
  });
  const request = mock(async () => new Response('{}'));
  globalThis.fetch = request as unknown as typeof fetch;
  let session!: ReturnType<typeof useCaptureRecordingSession>;
  function Capture() {
    session = useCaptureRecordingSession();
    return null;
  }
  const platform: Pick<Platform, 'metadata' | 'events'> = {
    metadata: { isTauri: false, getVersion: async () => 'test' },
    events: { emit: async () => {}, subscribe: () => () => {} },
  };
  try {
    await act(async () => {
      renderer = create(
        <QueryClientProvider client={queryClient}>
          <PlatformProvider platform={platform as Platform}>
            <Capture />
          </PlatformProvider>
        </QueryClientProvider>,
      );
    });
    await act(async () => {
      session.startRecording();
    });
    expect(getUserMedia).toHaveBeenCalledTimes(1);
    useServerStore.getState().setServerUrl('https://new.example');
    useServerStore.getState().setRemoteApiToken('new-server-token');
    now = 2000;
    recorder.ondataavailable?.({ data: new Blob(['synthetic audio']) });
    await act(async () => {
      session.stopRecording();
      await recorder.onstop?.();
    });
    expect(request).not.toHaveBeenCalled();
    expect(session.errorMessage).toContain('Connection changed');
    expect(trackStop).toHaveBeenCalledTimes(1);
  } finally {
    act(() => {
      renderer?.unmount();
      renderer = undefined;
    });
    globalThis.MediaRecorder = originalRecorder;
    globalThis.fetch = originalFetch;
    Date.now = originalDateNow;
    queryClient.clear();
  }
});

test('story playback streams only active clips and releases them on pause', async () => {
  useServerStore.setState({ serverUrl: 'http://127.0.0.1:17493', remoteApiToken: '' });
  const players: FakeAudio[] = [];
  const frames = new Map<number, FrameRequestCallback>();
  let nextFrame = 0;
  let now = 0;
  class FakeAudio {
    src = '';
    currentTime = 0;
    crossOrigin = '';
    preload = '';
    onloadedmetadata: (() => void) | null = null;
    onended: (() => void) | null = null;
    onerror: (() => void) | null = null;
    pause = mock(() => {});
    play = mock(async () => {});
    constructor() {
      players.push(this);
    }
    removeAttribute() {
      this.src = '';
    }
    load() {
      if (this.src) this.onloadedmetadata?.();
    }
  }
  const connection = () => ({ connect() {}, disconnect() {}, gain: { value: 1 } });
  class FakeContext {
    state = 'running';
    sampleRate = 48000;
    destination = {};
    get currentTime() {
      return now;
    }
    createGain = connection;
    createMediaElementSource = connection;
    close = async () => {
      this.state = 'closed';
    };
    decodeAudioData() {
      throw new Error('Must not buffer and decode a full story');
    }
  }
  Object.assign(globalThis, {
    Audio: FakeAudio,
    AudioContext: FakeContext,
    requestAnimationFrame: (callback: FrameRequestCallback) => {
      frames.set(++nextFrame, callback);
      return nextFrame;
    },
    cancelAnimationFrame: (id: number) => {
      frames.delete(id);
    },
  });
  const clips = Array.from({ length: 100 }, (_, index) => ({
    id: `clip-${index}`,
    generation_id: `generation-${index}`,
    duration: 3600,
    start_time_ms: index * 3600_000,
    trim_start_ms: 0,
    trim_end_ms: 0,
  })) as StoryItemDetail[];
  function Playback() {
    useStoryPlayback();
    return null;
  }
  act(() => {
    renderer = create(createElement(Playback));
  });
  expect(players).toHaveLength(0);
  await act(async () => useStoryStore.getState().play('story', clips));
  expect(players).toHaveLength(1);
  expect(players[0].src).toContain('generation-0');
  expect(players[0].play).toHaveBeenCalledTimes(1);
  now = 1;
  act(() => {
    const callbacks = [...frames.values()];
    frames.clear();
    callbacks.forEach((callback) => {
      callback(1000);
    });
  });
  expect(useStoryStore.getState().currentTimeMs).toBe(1000);
  await act(async () => useStoryStore.getState().seek(3600_000));
  expect(players[0].src).toBe('');
  expect(players.at(-1)?.src).toContain('generation-1');
  act(() => useStoryStore.getState().pause());
  expect(players.every((player) => player.src === '')).toBe(true);
});
