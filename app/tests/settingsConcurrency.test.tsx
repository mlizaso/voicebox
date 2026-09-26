import { afterEach, beforeEach, expect, mock, spyOn, test } from 'bun:test';
import { QueryClientProvider } from '@tanstack/react-query';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import { apiClient } from '../src/lib/api/client';
import type { CaptureSettings, GenerationSettings } from '../src/lib/api/types';
import { useCaptureSettings, useGenerationSettings } from '../src/lib/hooks/useSettings';
import { queryClient } from '../src/lib/queryClient';
import { useServerStore } from '../src/stores/serverStore';

let renderer: ReactTestRenderer | undefined;
const originalSettings: GenerationSettings = {
  max_chunk_chars: 800,
  crossfade_ms: 50,
  normalize_audio: true,
  autoplay_on_generate: true,
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function connect(connectionId: string) {
  return useServerStore.getState().setConnection({
    connectionId,
    serverUrl: 'http://127.0.0.1:17493',
    remoteApiToken: '',
    mode: 'local',
  });
}

beforeEach(() => {
  queryClient.clear();
  connect('settings-original');
  spyOn(apiClient, 'getGenerationSettings').mockResolvedValue(originalSettings);
});

afterEach(() => {
  act(() => renderer?.unmount());
  renderer = undefined;
  mock.restore();
  queryClient.clear();
});

async function mountSettings() {
  const hooks: ReturnType<typeof useGenerationSettings>[] = [];
  function Settings({ index }: { index: number }) {
    hooks[index] = useGenerationSettings();
    return null;
  }
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={queryClient}>
        <Settings index={0} />
        <Settings index={1} />
      </QueryClientProvider>,
    );
  });
  return hooks;
}

// React Query schedules observer notifications outside the mutation promise.
async function flush() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

async function waitForMutations(count: number) {
  if (queryClient.isMutating() !== count) {
    await new Promise<void>((resolve) => {
      const unsubscribe = queryClient.getMutationCache().subscribe(() => {
        if (queryClient.isMutating() === count) {
          unsubscribe();
          resolve();
        }
      });
    });
  }
  await flush();
}

async function waitForSettingsRead() {
  const query = queryClient.getQueryCache().find({ queryKey: ['settings', 'generation'] });
  if (query?.state.fetchStatus === 'fetching') await query.promise;
  await flush();
}

test('settings writes serialize across hook instances without erasing newer optimistic values', async () => {
  const first = deferred<GenerationSettings>();
  const second = deferred<GenerationSettings>();
  const save = spyOn(apiClient, 'updateGenerationSettings')
    .mockImplementationOnce(() => first.promise)
    .mockImplementationOnce(() => second.promise);
  const hooks = await mountSettings();
  await act(async () => {
    hooks[0].update({ normalize_audio: false });
    await flush();
  });
  await act(async () => {
    hooks[1].update({ normalize_audio: true, crossfade_ms: 100 });
    await flush();
  });
  expect(hooks[0].settings).toEqual({ ...originalSettings, crossfade_ms: 100 });
  expect(save).toHaveBeenCalledTimes(1);
  await act(async () => {
    first.resolve({ ...originalSettings, normalize_audio: false });
    await waitForMutations(1);
  });
  expect(hooks[0].settings).toEqual({ ...originalSettings, crossfade_ms: 100 });
  expect(save).toHaveBeenCalledTimes(2);
  await act(async () => {
    second.resolve({ ...originalSettings, crossfade_ms: 100 });
    await waitForMutations(0);
  });
  expect(hooks[1].settings).toEqual({ ...originalSettings, crossfade_ms: 100 });
  expect(queryClient.getQueryData<GenerationSettings>(['settings', 'generation'])).toEqual(
    hooks[1].settings,
  );
});

test('failed settings writes remove only their own optimistic patch', async () => {
  const first = deferred<GenerationSettings>();
  const second = deferred<GenerationSettings>();
  spyOn(apiClient, 'updateGenerationSettings')
    .mockImplementationOnce(() => first.promise)
    .mockImplementationOnce(() => second.promise);
  const hooks = await mountSettings();
  await act(async () => {
    hooks[0].update({ normalize_audio: false });
    await flush();
  });
  await act(async () => {
    hooks[1].update({ crossfade_ms: 100 });
    await flush();
  });
  await act(async () => {
    first.reject(new Error('First update failed'));
    await waitForMutations(1);
  });
  expect(hooks[0].settings).toEqual({ ...originalSettings, crossfade_ms: 100 });
  await act(async () => {
    second.reject(new Error('Second update failed'));
    await waitForMutations(0);
  });
  expect(hooks[0].settings).toEqual(originalSettings);
  expect(queryClient.getQueryData<GenerationSettings>(['settings', 'generation'])).toEqual(
    originalSettings,
  );
});

test('pending settings writes retain the connection lock and release it after completion', async () => {
  const saved = deferred<GenerationSettings>();
  spyOn(apiClient, 'updateGenerationSettings').mockImplementation(() => saved.promise);
  const hooks = await mountSettings();
  await act(async () => {
    hooks[0].update({ normalize_audio: false });
    await flush();
  });
  expect(connect('settings-replacement')).toBe(false);
  expect(useServerStore.getState().connectionId).toBe('settings-original');
  await act(async () => {
    saved.resolve({ ...originalSettings, normalize_audio: false });
    await waitForMutations(0);
  });
  await act(async () => {
    expect(connect('settings-replacement')).toBe(true);
    await flush();
  });
  expect(hooks[0].settings).toEqual(originalSettings);
});

test('a retained settings callback cannot write or cancel the new connection initial read', async () => {
  const save = spyOn(apiClient, 'updateGenerationSettings');
  const replacementRead = deferred<GenerationSettings>();
  const read = spyOn(apiClient, 'getGenerationSettings')
    .mockResolvedValueOnce(originalSettings)
    .mockImplementation(() => replacementRead.promise);
  const hooks = await mountSettings();
  const oldUpdate = hooks[0].update;
  await act(async () => {
    connect('settings-replacement');
    await flush();
  });
  expect(read).toHaveBeenCalledTimes(2);
  expect(hooks[0].settings).toBeUndefined();
  await act(async () => {
    oldUpdate({ normalize_audio: false });
    await waitForMutations(0);
  });
  expect(save).not.toHaveBeenCalled();
  const replacementSettings = { ...originalSettings, max_chunk_chars: 1000 };
  await act(async () => {
    replacementRead.resolve(replacementSettings);
    await waitForSettingsRead();
  });
  expect(hooks[0].settings).toEqual(replacementSettings);
});

test('a failed write before initial settings arrive restarts the cancelled read', async () => {
  const initialRead = deferred<GenerationSettings>();
  const recoveredSettings = { ...originalSettings, max_chunk_chars: 1000 };
  const read = spyOn(apiClient, 'getGenerationSettings')
    .mockImplementationOnce(() => initialRead.promise)
    .mockResolvedValueOnce(recoveredSettings);
  spyOn(apiClient, 'updateGenerationSettings').mockRejectedValue(new Error('Save failed'));
  const hooks = await mountSettings();
  expect(hooks[0].settings).toBeUndefined();
  await act(async () => {
    hooks[0].update({ normalize_audio: false });
    await waitForMutations(0);
  });
  expect(read).toHaveBeenCalledTimes(2);
  expect(hooks[0].settings).toEqual(recoveredSettings);
  await act(async () => {
    initialRead.resolve(originalSettings);
    await flush();
  });
  expect(hooks[0].settings).toEqual(recoveredSettings);
});

test.each([
  'success',
  'failure',
])('initial-read recovery waits for a queued write ending in %s', async (outcome) => {
  const initialRead = deferred<GenerationSettings>();
  const first = deferred<GenerationSettings>();
  const second = deferred<GenerationSettings>();
  const confirmedSettings = { ...originalSettings, crossfade_ms: 100 };
  const read = spyOn(apiClient, 'getGenerationSettings')
    .mockImplementationOnce(() => initialRead.promise)
    .mockResolvedValueOnce(originalSettings);
  spyOn(apiClient, 'updateGenerationSettings')
    .mockImplementationOnce(() => first.promise)
    .mockImplementationOnce(() => second.promise);
  const hooks = await mountSettings();
  await act(async () => {
    hooks[0].update({ normalize_audio: false });
    await flush();
  });
  await act(async () => {
    hooks[1].update({ crossfade_ms: 100 });
    await flush();
  });
  await act(async () => {
    first.reject(new Error('First write failed'));
    await waitForMutations(1);
  });
  expect(read).toHaveBeenCalledTimes(1);
  await act(async () => {
    if (outcome === 'success') second.resolve(confirmedSettings);
    else second.reject(new Error('Second write failed'));
    await waitForMutations(0);
  });
  expect(read).toHaveBeenCalledTimes(outcome === 'success' ? 1 : 2);
  expect(hooks[0].settings).toEqual(outcome === 'success' ? confirmedSettings : originalSettings);
  await act(async () => {
    initialRead.resolve({ ...originalSettings, crossfade_ms: 1 });
    await flush();
  });
  expect(hooks[0].settings).toEqual(outcome === 'success' ? confirmedSettings : originalSettings);
});

test('a delayed recovery read cannot overwrite a subsequent successful settings write', async () => {
  const initialRead = deferred<GenerationSettings>();
  const recoveryRead = deferred<GenerationSettings>();
  const confirmedSettings = { ...originalSettings, crossfade_ms: 100 };
  const read = spyOn(apiClient, 'getGenerationSettings')
    .mockImplementationOnce(() => initialRead.promise)
    .mockImplementationOnce(() => recoveryRead.promise);
  spyOn(apiClient, 'updateGenerationSettings')
    .mockRejectedValueOnce(new Error('First write failed'))
    .mockResolvedValueOnce(confirmedSettings);
  const hooks = await mountSettings();
  await act(async () => {
    hooks[0].update({ normalize_audio: false });
    await waitForMutations(0);
  });
  expect(read).toHaveBeenCalledTimes(2);
  await act(async () => {
    hooks[1].update({ crossfade_ms: 100 });
    await waitForMutations(0);
  });
  expect(hooks[0].settings).toEqual(confirmedSettings);
  await act(async () => {
    recoveryRead.resolve(originalSettings);
    initialRead.resolve(originalSettings);
    await flush();
  });
  expect(hooks[0].settings).toEqual(confirmedSettings);
});

test('settings completion callbacks retain the submitted public patch', async () => {
  spyOn(apiClient, 'updateGenerationSettings').mockResolvedValue({
    ...originalSettings,
    normalize_audio: false,
  });
  const hooks = await mountSettings();
  const success = mock((_data: GenerationSettings, _patch: Partial<GenerationSettings>) => {});
  await act(async () => {
    hooks[0].update({ normalize_audio: false }, { onSuccess: success });
    await waitForMutations(0);
  });
  expect(success).toHaveBeenCalledTimes(1);
  expect(success.mock.calls[0][1]).toEqual({ normalize_audio: false });
});

test('capture model changes still invalidate readiness after a settings write', async () => {
  const initial: CaptureSettings = {
    stt_model: 'base',
    language: 'en',
    auto_refine: true,
    llm_model: '0.6B',
    smart_cleanup: true,
    self_correction: true,
    preserve_technical: true,
    allow_auto_paste: false,
    default_playback_voice_id: null,
    hotkey_enabled: false,
    chord_push_to_talk_keys: [],
    chord_toggle_to_talk_keys: [],
  };
  spyOn(apiClient, 'getCaptureSettings').mockResolvedValue(initial);
  spyOn(apiClient, 'updateCaptureSettings').mockResolvedValue({ ...initial, stt_model: 'small' });
  const invalidate = spyOn(queryClient, 'invalidateQueries');
  let settings!: ReturnType<typeof useCaptureSettings>;
  function Capture() {
    settings = useCaptureSettings();
    return null;
  }
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={queryClient}>
        <Capture />
      </QueryClientProvider>,
    );
  });
  await act(async () => {
    settings.update({ stt_model: 'small' });
    await flush();
  });
  expect(settings.settings?.stt_model).toBe('small');
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ['capture-readiness'] });
});
