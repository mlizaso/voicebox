import { afterEach, expect, mock, test } from 'bun:test';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import { useAudioDuration } from '../src/lib/hooks/useAudioDuration';
import { useAudioPlayer } from '../src/lib/hooks/useAudioPlayer';
import { useSystemAudioCapture } from '../src/lib/hooks/useSystemAudioCapture';
import { convertToWav, getAudioDuration } from '../src/lib/utils/audio';
import { PlatformProvider } from '../src/platform/PlatformContext';
import type { Platform } from '../src/platform/types';

const originalAudio = globalThis.Audio;
const originalContext = globalThis.AudioContext;
const originalWindow = globalThis.window;
const originalCreateUrl = URL.createObjectURL;
const originalRevokeUrl = URL.revokeObjectURL;
const originalNow = Date.now;
let renderer: ReactTestRenderer | undefined;

afterEach(() => {
  act(() => renderer?.unmount());
  renderer = undefined;
  Object.assign(globalThis, {
    Audio: originalAudio,
    AudioContext: originalContext,
    window: originalWindow,
  });
  URL.createObjectURL = originalCreateUrl;
  URL.revokeObjectURL = originalRevokeUrl;
  Date.now = originalNow;
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function mountPlayer(play: () => Promise<void> = async () => {}) {
  const players: FakeAudio[] = [];
  class FakeAudio extends EventTarget {
    pause = mock(() => {});
    play = mock(play);
    constructor(public src: string) {
      super();
      players.push(this);
    }
    removeAttribute() {
      this.src = '';
    }
    load() {}
  }
  globalThis.Audio = FakeAudio as unknown as typeof Audio;
  URL.createObjectURL = mock((blob: Blob) => `blob:${(blob as File).name}`);
  const revoke = mock(() => {});
  URL.revokeObjectURL = revoke;
  let player!: ReturnType<typeof useAudioPlayer>;
  function Player() {
    player = useAudioPlayer();
    return null;
  }
  act(() => {
    renderer = create(<Player />);
  });
  return {
    players,
    revoke,
    get player() {
      return player;
    },
  };
}

test('previewing a replacement file releases the old audio and plays the selected file', async () => {
  const preview = mountPlayer();
  const first = new File(['a'], 'a.wav');
  const second = new File(['b'], 'b.wav');
  await act(async () => preview.player.playPause(first));
  act(() => preview.player.playPause(first));
  await act(async () => preview.player.playPause(second));
  expect(preview.players).toHaveLength(2);
  expect(preview.players[1].src).toBe('blob:b.wav');
  expect(preview.players[1].play).toHaveBeenCalledTimes(1);
  expect(preview.revoke).toHaveBeenCalledWith('blob:a.wav');
  act(() => renderer?.unmount());
  renderer = undefined;
  expect(preview.players[1].pause).toHaveBeenCalledTimes(1);
  expect(preview.revoke).toHaveBeenCalledWith('blob:b.wav');
});

test('a rejected play request clears playback state and releases its object URL', async () => {
  const preview = mountPlayer(async () => {
    throw new Error('Playback denied');
  });
  await act(async () => preview.player.playPause(new File(['a'], 'a.wav')));
  expect(preview.player.isPlaying).toBe(false);
  expect(preview.revoke).toHaveBeenCalledWith('blob:a.wav');
});

test('an old play rejection cannot clear a replacement preview', async () => {
  const firstPlay = deferred<void>();
  let attempt = 0;
  const preview = mountPlayer(() => (++attempt === 1 ? firstPlay.promise : Promise.resolve()));
  act(() => preview.player.playPause(new File(['a'], 'a.wav')));
  await act(async () => preview.player.playPause(new File(['b'], 'b.wav')));
  await act(async () => firstPlay.reject(new Error('Old request cancelled')));
  expect(preview.player.isPlaying).toBe(true);
  expect(preview.revoke).not.toHaveBeenCalledWith('blob:b.wav');
});

test('WAV conversion closes its decoding context when decoding fails', async () => {
  const close = mock(async () => {});
  globalThis.AudioContext = class {
    close = close;
    async decodeAudioData() {
      throw new Error('Unsupported codec');
    }
  } as unknown as typeof AudioContext;
  await expect(convertToWav(new Blob(['audio']))).rejects.toThrow('Unsupported codec');
  expect(close).toHaveBeenCalledTimes(1);
});

test('duration decoding and WAV encoding preserve decoded audio and release their contexts', async () => {
  const close = mock(async () => {});
  const decoded = {
    duration: 0.5,
    numberOfChannels: 1,
    sampleRate: 4,
    length: 2,
    getChannelData: () => new Float32Array([-1, 1]),
  };
  globalThis.AudioContext = class {
    close = close;
    async decodeAudioData() {
      return decoded;
    }
  } as unknown as typeof AudioContext;
  expect(await getAudioDuration(new File(['audio'], 'a.wav'))).toBe(0.5);
  const wav = new DataView(await (await convertToWav(new Blob(['audio']))).arrayBuffer());
  expect(wav.byteLength).toBe(48);
  expect(wav.getUint32(24, true)).toBe(4);
  expect(wav.getInt16(44, true)).toBe(-32768);
  expect(wav.getInt16(46, true)).toBe(32767);
  expect(close).toHaveBeenCalledTimes(2);
});

test('duration validation ignores an older file that finishes after its replacement', async () => {
  const decodes = [deferred<AudioBuffer>(), deferred<AudioBuffer>()];
  let nextDecode = 0;
  const close = mock(async () => {});
  globalThis.AudioContext = class {
    close = close;
    decodeAudioData() {
      return decodes[nextDecode++].promise;
    }
  } as unknown as typeof AudioContext;
  let result!: ReturnType<typeof useAudioDuration>;
  function Duration({ file }: { file?: File }) {
    result = useAudioDuration(file);
    return null;
  }
  await act(async () => {
    renderer = create(<Duration file={new File(['a'], 'a.wav')} />);
  });
  const replacement = new File(['b'], 'b.wav');
  await act(async () => renderer?.update(<Duration file={replacement} />));
  expect(result.isLoading).toBe(true);
  await act(async () => decodes[1].resolve({ duration: 10 } as AudioBuffer));
  expect(result.duration).toBe(10);
  await act(async () => decodes[0].resolve({ duration: 90 } as AudioBuffer));
  expect(result.duration).toBe(10);
  expect(result.isLoading).toBe(false);
  expect(close).toHaveBeenCalledTimes(2);
  act(() => renderer?.update(<Duration />));
  expect(result.duration).toBeNull();
  expect(result.isLoading).toBe(false);
});

async function mountSystemCapture(startCapture: () => Promise<void> = async () => {}) {
  const stopped = deferred<Blob>();
  const start = mock(startCapture);
  const stop = mock(() => stopped.promise);
  const complete = mock((_blob: Blob, _duration?: number) => {});
  Object.assign(globalThis, { window: { setInterval, clearInterval } });
  let capture!: ReturnType<typeof useSystemAudioCapture>;
  function Capture() {
    capture = useSystemAudioCapture({ onRecordingComplete: complete });
    return null;
  }
  const platform = {
    metadata: { isTauri: true },
    audio: {
      isSystemAudioSupported: async () => true,
      startSystemAudioCapture: start,
      stopSystemAudioCapture: stop,
    },
  } as unknown as Platform;
  await act(async () => {
    renderer = create(
      <PlatformProvider platform={platform}>
        <Capture />
      </PlatformProvider>,
    );
  });
  return {
    start,
    stop,
    complete,
    stopped,
    get capture() {
      return capture;
    },
  };
}

test.each([
  'cancel',
  'unmount',
])('system capture suppresses pending completion after %s', async (action) => {
  const session = await mountSystemCapture();
  await act(async () => session.capture.startRecording());
  let stopping!: Promise<void>;
  act(() => {
    stopping = session.capture.stopRecording();
  });
  act(() => {
    if (action === 'cancel') void session.capture.cancelRecording();
    else {
      renderer?.unmount();
      renderer = undefined;
    }
  });
  await act(async () => {
    session.stopped.resolve(new Blob(['audio']));
    await stopping;
  });
  expect(session.stop).toHaveBeenCalledTimes(1);
  expect(session.complete).not.toHaveBeenCalled();
});

test('system capture waits for native stop and measures duration at stop request', async () => {
  let now = 1000;
  Date.now = () => now;
  const session = await mountSystemCapture();
  await act(async () => session.capture.startRecording());
  now = 3000;
  let stopping!: Promise<void>;
  act(() => {
    stopping = session.capture.stopRecording();
  });
  await act(async () => session.capture.startRecording());
  expect(session.start).toHaveBeenCalledTimes(1);
  now = 10000;
  const blob = new Blob(['audio']);
  await act(async () => {
    session.stopped.resolve(blob);
    await stopping;
  });
  expect(session.complete).toHaveBeenCalledWith(blob, 2);
});

test('cancelling before native start settles releases the late device without recording', async () => {
  const starting = deferred<void>();
  const session = await mountSystemCapture(() => starting.promise);
  let pendingStart!: Promise<void>;
  act(() => {
    pendingStart = session.capture.startRecording();
  });
  await act(async () => session.capture.cancelRecording());
  expect(session.capture.isRecording).toBe(false);
  await act(async () => {
    starting.resolve();
  });
  expect(session.stop).toHaveBeenCalledTimes(1);
  await act(async () => {
    session.stopped.resolve(new Blob(['discarded']));
    await pendingStart;
  });
  expect(session.capture.isRecording).toBe(false);
  expect(session.complete).not.toHaveBeenCalled();
});
