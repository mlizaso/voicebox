import { afterEach, beforeEach, expect, mock, spyOn, test } from 'bun:test';
import { defaultScheduler, notifyManager, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { I18nextProvider } from 'react-i18next';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import WaveSurfer from 'wavesurfer.js';
import WebAudioPlayer from 'wavesurfer.js/dist/webaudio.js';
import { AudioPlayer } from '../src/components/AudioPlayer/AudioPlayer';
import { EffectsDetail } from '../src/components/EffectsTab/EffectsDetail';
import { apiClient } from '../src/lib/api/client';
import type { HistoryResponse } from '../src/lib/api/types';
import { queryClient } from '../src/lib/queryClient';
import { PlatformProvider } from '../src/platform/PlatformContext';
import type { Platform } from '../src/platform/types';
import { useEffectsStore } from '../src/stores/effectsStore';
import { usePlayerStore } from '../src/stores/playerStore';
import { useServerStore } from '../src/stores/serverStore';
import { testI18n } from './testI18n';

const originalWindow = globalThis.window;
const originalDocument = globalThis.document;
const originalStyle = globalThis.getComputedStyle;
const originalFrame = globalThis.requestAnimationFrame;
const originalCancelFrame = globalThis.cancelAnimationFrame;
const originalFetch = globalThis.fetch;
const originalAudioContext = globalThis.AudioContext;
const originalTimeout = globalThis.setTimeout;
const originalClearTimeout = globalThis.clearTimeout;
let renderer: ReactTestRenderer | undefined;
let visible = true;
const frames = new Map<number, FrameRequestCallback>();
let nextFrame = 0;
const instances: FakeWaveSurfer[] = [];

class FakeWaveSurfer {
  events = new Map<string, (...args: never[]) => void>();
  playing = false;
  currentTime = 0;
  constructor(readonly container: unknown) {}
  on = (event: string, callback: (...args: never[]) => void) => {
    this.events.set(event, callback);
    return () => this.events.delete(event);
  };
  emit(event: string) {
    this.events.get(event)?.();
  }
  load = mock(async (_url: string) => {
    this.emit('ready');
  });
  destroy = mock(() => {});
  unAll = mock(() => {
    this.events.clear();
  });
  pause = mock(() => {
    const wasPlaying = this.playing;
    this.playing = false;
    if (wasPlaying) this.emit('pause');
  });
  play = mock(async () => {
    this.playing = true;
    this.emit('play');
  });
  seekTo = mock((_progress: number) => {});
  setVolume = mock((_volume: number) => {});
  setMuted = mock((_muted: boolean) => {});
  setOptions = mock((_options: Parameters<WaveSurfer['setOptions']>[0]) => {});
  isPlaying = () => this.playing;
  getCurrentTime = () => this.currentTime;
  getDuration = () => 1;
  getMediaElement() {
    return {};
  }
}

beforeEach(() => {
  notifyManager.setScheduler(queueMicrotask);
  queryClient.clear();
  usePlayerStore.getState().reset();
  useServerStore.setState({
    connectionId: 'player-original',
    serverUrl: 'http://127.0.0.1:17493',
    remoteApiToken: '',
  });
  frames.clear();
  nextFrame = 0;
  visible = true;
  instances.length = 0;
  const getStyle = () => ({ display: 'block', visibility: 'visible', getPropertyValue: () => '' });
  Object.assign(globalThis, {
    window: {
      location: { origin: 'http://127.0.0.1:17493' },
      getComputedStyle: getStyle,
      setTimeout,
      clearTimeout,
    },
    document: { documentElement: {}, addEventListener() {}, removeEventListener() {} },
    getComputedStyle: getStyle,
    requestAnimationFrame: (callback: FrameRequestCallback) => {
      frames.set(++nextFrame, callback);
      return nextFrame;
    },
    cancelAnimationFrame: (id: number) => {
      frames.delete(id);
    },
  });
  spyOn(WaveSurfer, 'create').mockImplementation(({ container }) => {
    const instance = new FakeWaveSurfer(container);
    instances.push(instance);
    return instance as unknown as WaveSurfer;
  });
});

afterEach(() => {
  act(() => renderer?.unmount());
  renderer = undefined;
  usePlayerStore.getState().reset();
  queryClient.clear();
  mock.restore();
  notifyManager.setScheduler(defaultScheduler);
  Object.assign(globalThis, {
    window: originalWindow,
    document: originalDocument,
    getComputedStyle: originalStyle,
    requestAnimationFrame: originalFrame,
    cancelAnimationFrame: originalCancelFrame,
    fetch: originalFetch,
    AudioContext: originalAudioContext,
    setTimeout: originalTimeout,
    clearTimeout: originalClearTimeout,
  });
});

async function mount(
  children: ReactNode = <AudioPlayer />,
  platform = { metadata: { isTauri: false } } as Platform,
) {
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={queryClient}>
        <PlatformProvider platform={platform}>
          <I18nextProvider i18n={testI18n}>{children}</I18nextProvider>
        </PlatformProvider>
      </QueryClientProvider>,
      {
        createNodeMock: (element) =>
          element.props.className === 'w-full min-h-[80px] select-none'
            ? { getBoundingClientRect: () => ({ width: visible ? 640 : 0, height: 80 }) }
            : null,
      },
    );
  });
}

async function flushFrames() {
  await act(async () => {
    const callbacks = [...frames.values()];
    frames.clear();
    for (const callback of callbacks) callback(0);
  });
}

test('closing and reopening creates a waveform in the new container while track changes reuse it', async () => {
  usePlayerStore.getState().setAudio('http://127.0.0.1:17493/audio/a', 'a', null);
  await mount();
  await flushFrames();
  expect(instances).toHaveLength(1);
  expect(instances[0].load).toHaveBeenCalledWith('http://127.0.0.1:17493/audio/a');
  await act(async () =>
    usePlayerStore.getState().setAudio('http://127.0.0.1:17493/audio/b', 'b', null),
  );
  await flushFrames();
  expect(instances).toHaveLength(1);
  expect(instances[0].load).toHaveBeenCalledWith('http://127.0.0.1:17493/audio/b');
  act(() => renderer!.root.findByProps({ 'aria-label': 'Close player' }).props.onClick());
  expect(renderer!.toJSON()).toBeNull();
  expect(instances[0].destroy).toHaveBeenCalledTimes(1);
  await act(async () =>
    usePlayerStore.getState().setAudio('http://127.0.0.1:17493/audio/c', 'c', null),
  );
  await flushFrames();
  expect(instances).toHaveLength(2);
  expect(instances[1].container).not.toBe(instances[0].container);
  expect(instances[1].load).toHaveBeenCalledWith('http://127.0.0.1:17493/audio/c');
});

test('successive real waveform loads receive fresh live signals and cancel their previous load', async () => {
  Object.assign(window, { AbortController });
  Object.assign(globalThis, {
    AudioContext: class {
      state = 'running';
      currentTime = 0;
      destination = {};
      createGain() {
        return { context: this, gain: { value: 1 }, connect() {}, disconnect() {} };
      }
      async decodeAudioData() {
        return { duration: 1 };
      }
      async close() {
        this.state = 'closed';
      }
    },
  });
  const requests: Array<{
    signal: AbortSignal | null | undefined;
    abortedAtFetch: boolean;
    credentials: RequestCredentials | undefined;
  }> = [];
  spyOn(globalThis, 'fetch').mockImplementation(((url, options) => {
    if (options) {
      requests.push({
        signal: options.signal,
        abortedAtFetch: options.signal?.aborted ?? false,
        credentials: options.credentials,
      });
    }
    return originalFetch(url, options);
  }) as typeof fetch);
  const createWaveform = spyOn(WaveSurfer, 'create').mockImplementation((options) => {
    // Exercise the library's real load/options/source/cleanup methods without a DOM canvas.
    return Object.assign(Object.create(WaveSurfer.prototype), {
      options,
      media: new WebAudioPlayer(),
      isExternalMedia: true,
      listeners: {},
      plugins: [],
      subscriptions: [],
      mediaSubscriptions: [],
      reactiveCleanups: [],
      reactiveMediaEventCleanups: [],
      timer: { destroy() {} },
      renderer: { setOptions() {}, render() {}, renderProgress() {}, destroy() {} },
    }) as WaveSurfer;
  });
  usePlayerStore.getState().setLocalAudioWithAutoPlay(new Blob(['first']), 'first');
  usePlayerStore.getState().clearAutoPlayFlag();
  await mount();
  await flushFrames();
  expect(requests).toHaveLength(1);
  expect(usePlayerStore.getState().duration).toBe(1);

  await act(async () => {
    usePlayerStore.getState().setLocalAudioWithAutoPlay(new Blob(['second']), 'second');
    usePlayerStore.getState().clearAutoPlayFlag();
  });
  expect(createWaveform).toHaveBeenCalledTimes(1);
  expect(requests).toHaveLength(2);
  expect(requests[1].signal).not.toBe(requests[0].signal);
  expect(requests[0].signal?.aborted).toBe(true);
  // Bun's native blob fetch accepts aborted signals, so inspect the request boundary.
  expect(requests.map((request) => request.abortedAtFetch)).toEqual([false, false]);
  expect(requests.map((request) => request.credentials)).toEqual(['omit', 'omit']);
  expect(usePlayerStore.getState().duration).toBe(1);
  act(() => usePlayerStore.getState().reset());
  expect(requests[1].signal?.aborted).toBe(true);
});

test.each([
  'connection change',
  'unmount',
])('playing audio stops on %s without the close button', async (end) => {
  usePlayerStore.getState().setAudioWithAutoPlay('http://127.0.0.1:17493/audio/a', 'a', null);
  await mount();
  await flushFrames();
  expect(instances[0].isPlaying()).toBe(true);
  instances[0].currentTime = 0.5;
  act(() => {
    if (end === 'connection change') {
      expect(
        useServerStore.getState().setConnection({
          connectionId: 'player-replacement',
          serverUrl: 'http://127.0.0.1:17494',
          remoteApiToken: '',
          mode: 'local',
        }),
      ).toBe(true);
    } else {
      renderer?.unmount();
      renderer = undefined;
    }
  });
  // WaveSurfer's WebAudio backend is external media: destroy alone does not pause it.
  expect(instances[0].destroy).toHaveBeenCalledTimes(1);
  expect(instances[0].isPlaying()).toBe(false);
  expect(usePlayerStore.getState().currentTime).toBe(end === 'connection change' ? 0 : 0.5);
});

test('a disposed waveform cannot consume autoplay or change duration after reopening', async () => {
  usePlayerStore.getState().setAudio('http://127.0.0.1:17493/audio/a', 'a', null);
  await mount();
  await flushFrames();
  const old = instances[0];
  act(() => usePlayerStore.getState().reset());
  act(() =>
    usePlayerStore.getState().setAudioWithAutoPlay('http://127.0.0.1:17493/audio/b', 'b', null),
  );
  act(() => old.emit('ready'));
  expect(usePlayerStore.getState().duration).toBe(0);
  expect(usePlayerStore.getState().shouldAutoPlay).toBe(true);
  expect(old.play).not.toHaveBeenCalled();
  await flushFrames();
  expect(instances[1].play).toHaveBeenCalledTimes(1);
});

test.each([
  'internal',
  'provided',
])('waveform disposal releases its context and %s URL ownership correctly', async (source) => {
  usePlayerStore.getState().setLocalAudioWithAutoPlay(new Blob(['preview']), 'preview');
  const ownedUrl = usePlayerStore.getState().audioUrl!;
  const internalUrl =
    source === 'internal' ? URL.createObjectURL(new Blob(['decoded source'])) : ownedUrl;
  const close = mock(async () => {});
  const disconnect = mock(() => {});
  const media = {
    src: internalUrl,
    removeAttribute: () => {
      media.src = '';
    },
    getGainNode: () => ({ context: { state: 'running', close }, disconnect }),
  };
  spyOn(FakeWaveSurfer.prototype, 'getMediaElement').mockReturnValue(media);
  const revoke = spyOn(URL, 'revokeObjectURL');
  try {
    await mount();
    await flushFrames();
    act(() => {
      renderer?.unmount();
      renderer = undefined;
    });
    expect(media.src).toBe('');
    expect(disconnect).toHaveBeenCalledTimes(1);
    expect(close).toHaveBeenCalledTimes(1);
    if (source === 'internal') {
      await expect(originalFetch(internalUrl)).rejects.toThrow();
      expect(revoke.mock.calls.filter(([url]) => url === internalUrl)).toHaveLength(1);
    }
    expect(await (await originalFetch(ownedUrl)).text()).toBe('preview');
    expect(revoke).not.toHaveBeenCalledWith(ownedUrl);
    act(() => usePlayerStore.getState().reset());
    expect(revoke.mock.calls.filter(([url]) => url === ownedUrl)).toHaveLength(1);
  } finally {
    if (source === 'internal') URL.revokeObjectURL(internalUrl);
  }
});

test.each(
  ['unchanged', 'close', 'reopen', 'replace', 'replace and return'].flatMap((transition) =>
    ['resolve', 'reject'].map((result) => ({ transition, result })),
  ),
)('native pause after $transition only affects its original playback when IPC will $result', async ({
  transition,
  result,
}) => {
  spyOn(apiClient, 'getProfileChannels').mockResolvedValue({ channel_ids: ['channel'] });
  spyOn(apiClient, 'listChannels').mockResolvedValue([
    { id: 'channel', name: 'Output', is_default: false, device_ids: ['device'], created_at: '' },
  ]);
  globalThis.fetch = (async () => new Response(new Uint8Array([1, 2]))) as unknown as typeof fetch;
  let settle!: () => void;
  const stopped = new Promise<void>((resolve, reject) => {
    settle = () => (result === 'resolve' ? resolve() : reject(new Error('IPC stop failed')));
  });
  const stopPlayback = mock(async () => {})
    .mockImplementationOnce(async () => {})
    .mockImplementationOnce(() => stopped);
  const playToDevices = mock(async () => {});
  const platform = {
    metadata: { isTauri: true },
    audio: { stopPlayback, playToDevices },
  } as unknown as Platform;
  const originalUrl = 'http://127.0.0.1:17493/audio/a';
  usePlayerStore.getState().setAudio(originalUrl, 'a', 'profile');
  await mount(<AudioPlayer />, platform);
  await flushFrames();
  await act(async () => {
    await renderer!.root.findByProps({ 'aria-label': 'Play' }).props.onClick();
  });
  expect(playToDevices).toHaveBeenCalledWith(new Uint8Array([1, 2]), ['device']);
  expect(instances[0].isPlaying()).toBe(true);
  let pauseResult!: Promise<unknown>;
  act(() => {
    pauseResult = renderer!.root
      .findByProps({ 'aria-label': 'Pause' })
      .props.onClick()
      .then(
        () => null,
        (error: unknown) => error,
      );
  });
  expect(stopPlayback).toHaveBeenCalledTimes(2);

  if (transition === 'close' || transition === 'reopen') {
    act(() => renderer!.root.findByProps({ 'aria-label': 'Close player' }).props.onClick());
    expect(instances[0].destroy).toHaveBeenCalledTimes(1);
  }
  if (transition === 'reopen' || transition.startsWith('replace')) {
    await act(async () =>
      usePlayerStore.getState().setAudioWithAutoPlay('http://127.0.0.1:17493/audio/b', 'b', null),
    );
    await flushFrames();
    if (transition === 'replace and return') {
      await act(async () =>
        usePlayerStore.getState().setAudioWithAutoPlay(originalUrl, 'a', 'profile'),
      );
    }
    expect(instances.at(-1)!.isPlaying()).toBe(true);
  }
  const player = instances.at(-1)!;
  const pauses = player.pause.mock.calls.length;
  await act(async () => {
    settle();
    expect(await pauseResult).toBeNull();
  });
  expect(player.pause).toHaveBeenCalledTimes(pauses + (transition === 'unchanged' ? 1 : 0));
  if (transition !== 'unchanged' && transition !== 'close') expect(player.isPlaying()).toBe(true);
});

test.each([
  'close',
  'unmount',
])('initialization retries are canceled and become inert after %s', async (end) => {
  visible = false;
  let retry!: () => void;
  const retryTimer = 0x5143 as unknown as ReturnType<typeof setTimeout>;
  const scheduled = mock(() => {});
  globalThis.setTimeout = ((callback: () => void, delay?: number) => {
    if (delay === 50) {
      scheduled();
      retry = callback;
      return retryTimer;
    }
    return originalTimeout(callback, delay);
  }) as typeof setTimeout;
  const clear = mock((timer: ReturnType<typeof setTimeout> | undefined) => {
    if (timer !== retryTimer) originalClearTimeout(timer);
  });
  globalThis.clearTimeout = clear as typeof clearTimeout;
  usePlayerStore.getState().setAudio('http://127.0.0.1:17493/audio/a', 'a', null);
  await mount();
  await flushFrames();
  expect(typeof retry).toBe('function');
  act(() => {
    if (end === 'close') usePlayerStore.getState().reset();
    else {
      renderer?.unmount();
      renderer = undefined;
    }
  });
  expect(clear).toHaveBeenCalledWith(retryTimer);
  act(() => {
    visible = true;
    retry();
  });
  expect(instances).toHaveLength(0);
  expect(scheduled).toHaveBeenCalledTimes(1);
});

async function mountPreview() {
  useEffectsStore.setState({
    isCreatingNew: true,
    selectedPresetId: null,
    workingChain: [{ type: 'gain', enabled: true, params: {} }],
  });
  spyOn(apiClient, 'listHistory').mockResolvedValue({
    items: [
      {
        id: 'generation',
        status: 'completed',
        text: 'hello',
        profile_name: 'Voice',
      } as HistoryResponse,
    ],
    total: 1,
  });
  spyOn(apiClient, 'getAvailableEffects').mockResolvedValue({ effects: [] });
  spyOn(apiClient, 'listEffectPresets').mockResolvedValue([]);
  await mount(
    <>
      <EffectsDetail />
      <AudioPlayer />
    </>,
  );
  return renderer!.root
    .findAllByType('button')
    .find((button) => button.children.includes('effects.preview.button'))!;
}

test('an oversized local effects preview cannot replace or revoke the current audio', async () => {
  usePlayerStore.getState().setLocalAudioWithAutoPlay(new Blob(['current']), 'current');
  const currentUrl = usePlayerStore.getState().audioUrl!;
  spyOn(apiClient, 'previewEffects').mockResolvedValue(
    new Blob([new Uint8Array(32 * 1024 * 1024 + 1)]),
  );
  const preview = await mountPreview();
  await flushFrames();
  const createUrl = spyOn(URL, 'createObjectURL');
  const revoke = spyOn(URL, 'revokeObjectURL');
  await act(async () => {
    await preview.props.onClick();
  });
  expect(usePlayerStore.getState().audioUrl).toBe(currentUrl);
  expect(createUrl).not.toHaveBeenCalled();
  expect(revoke).not.toHaveBeenCalledWith(currentUrl);
  expect(instances[0].load).toHaveBeenCalledTimes(1);
});

test('the local audio byte limit includes exactly 32 MiB', () => {
  usePlayerStore
    .getState()
    .setLocalAudioWithAutoPlay(new Blob([new Uint8Array(32 * 1024 * 1024)]), 'limit');
  expect(usePlayerStore.getState().audioId).toBe('limit');
});

test.each([
  'declared',
  'streamed',
])('effects preview downloads reject %s bodies exceeding 32 MiB', async (length) => {
  const cancel = mock(() => {});
  const chunk = new Uint8Array(1024 * 1024);
  let chunks = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (chunks++ < 40) controller.enqueue(chunk);
      else controller.close();
    },
    cancel,
  });
  globalThis.fetch = (async () =>
    new Response(body, {
      headers:
        length === 'declared'
          ? { 'content-length': String(33 * 1024 * 1024), 'content-type': 'audio/wav' }
          : { 'content-type': 'audio/wav' },
    })) as unknown as typeof fetch;
  await expect(apiClient.previewEffects('generation', [])).rejects.toThrow(
    'exceeds the allowed size',
  );
  expect(cancel).toHaveBeenCalledTimes(1);
});

test('effects previews reach the main player without network and release only when replaced or closed', async () => {
  const request = mock(() => Promise.reject(new Error('Unexpected network request')));
  globalThis.fetch = request as unknown as typeof fetch;
  spyOn(apiClient, 'previewEffects').mockResolvedValue(
    new Blob(['preview'], { type: 'audio/wav' }),
  );
  const revoke = spyOn(URL, 'revokeObjectURL');
  const preview = await mountPreview();
  await act(async () => {
    await preview.props.onClick();
  });
  const url = usePlayerStore.getState().audioUrl!;
  await flushFrames();
  expect(instances[0].load).toHaveBeenCalledWith(url);
  expect(instances[0].play).toHaveBeenCalledTimes(1);
  expect(await (await originalFetch(url)).text()).toBe('preview');
  expect(request).not.toHaveBeenCalled();
  await act(async () => {
    renderer!.update(
      <QueryClientProvider client={queryClient}>
        <PlatformProvider platform={{ metadata: { isTauri: false } } as Platform}>
          <AudioPlayer />
        </PlatformProvider>
      </QueryClientProvider>,
    );
  });
  expect(await (await originalFetch(url)).text()).toBe('preview');
  expect(revoke).not.toHaveBeenCalledWith(url);
  await act(async () =>
    usePlayerStore.getState().setAudio('http://127.0.0.1:17493/audio/b', 'b', null),
  );
  expect(revoke.mock.calls.filter(([revoked]) => revoked === url)).toHaveLength(1);
});

test('unmarked blob URLs cannot bypass the server audio boundary', async () => {
  const url = URL.createObjectURL(new Blob(['untrusted']));
  const request = mock(() => Promise.reject(new Error('Unexpected network request')));
  globalThis.fetch = request as unknown as typeof fetch;
  try {
    usePlayerStore.getState().setAudioWithAutoPlay(url, 'unmarked', null);
    await mount();
    await flushFrames();
    expect(instances[0].load).not.toHaveBeenCalled();
    expect(request).not.toHaveBeenCalled();
    expect(JSON.stringify(renderer!.toJSON())).toContain('connected server');
  } finally {
    URL.revokeObjectURL(url);
  }
});

test('closing a local preview revokes its URL exactly once', async () => {
  const revoke = spyOn(URL, 'revokeObjectURL');
  usePlayerStore.getState().setLocalAudioWithAutoPlay(new Blob(['preview']), 'preview');
  const url = usePlayerStore.getState().audioUrl!;
  await mount();
  await flushFrames();
  act(() => usePlayerStore.getState().reset());
  expect(revoke.mock.calls.filter(([revoked]) => revoked === url)).toHaveLength(1);
  await expect(originalFetch(url)).rejects.toThrow();
  act(() => usePlayerStore.getState().reset());
  expect(revoke.mock.calls.filter(([revoked]) => revoked === url)).toHaveLength(1);
});

test.each([
  'unmount',
  'connection change',
])('a late effects preview cannot restart playback after %s', async (end) => {
  let finish!: (blob: Blob) => void;
  const previewResult = new Promise<Blob>((resolve) => {
    finish = resolve;
  });
  spyOn(apiClient, 'previewEffects').mockImplementation(() => previewResult);
  const preview = await mountPreview();
  let pending!: Promise<void>;
  act(() => {
    pending = preview.props.onClick();
  });
  act(() => {
    if (end === 'unmount') {
      renderer?.unmount();
      renderer = undefined;
    } else
      useServerStore.getState().setConnection({
        connectionId: 'player-replacement',
        serverUrl: 'http://127.0.0.1:17494',
        remoteApiToken: '',
        mode: 'local',
      });
  });
  await act(async () => {
    finish(new Blob(['late preview']));
    await pending;
  });
  expect(usePlayerStore.getState().audioUrl).toBeNull();
  expect(instances).toHaveLength(0);
});
