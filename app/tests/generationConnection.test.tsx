import { afterEach, beforeEach, expect, mock, spyOn, test } from 'bun:test';
import { QueryClientProvider } from '@tanstack/react-query';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import { apiClient } from '../src/lib/api/client';
import type { GenerationResponse, ModelStatusListResponse } from '../src/lib/api/types';
import { useGenerationForm } from '../src/lib/hooks/useGenerationForm';
import { queryClient } from '../src/lib/queryClient';
import { useGenerationStore } from '../src/stores/generationStore';
import { useServerStore } from '../src/stores/serverStore';

let renderer: ReactTestRenderer | undefined;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function connect(connectionId: string) {
  useServerStore.getState().setConnection({
    connectionId,
    serverUrl: 'http://127.0.0.1:17493',
    remoteApiToken: '',
    mode: 'local',
  });
}

beforeEach(() => {
  queryClient.clear();
  connect('generation-original');
  spyOn(apiClient, 'getGenerationSettings').mockResolvedValue({
    max_chunk_chars: 800,
    crossfade_ms: 50,
    normalize_audio: true,
    autoplay_on_generate: false,
  });
});

afterEach(() => {
  act(() => renderer?.unmount());
  renderer = undefined;
  mock.restore();
  queryClient.clear();
  useGenerationStore.setState({ pendingGenerationIds: new Set(), isGenerating: false });
});

async function mountGeneration() {
  let generation!: ReturnType<typeof useGenerationForm>;
  const onSuccess = mock((_id: string) => {});
  function Generation() {
    generation = useGenerationForm({ onSuccess });
    return null;
  }
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={queryClient}>
        <Generation />
      </QueryClientProvider>,
    );
  });
  return {
    onSuccess,
    get generation() {
      return generation;
    },
  };
}

test('generation preflight cannot send original text to a newly selected connection', async () => {
  const status = deferred<ModelStatusListResponse>();
  spyOn(apiClient, 'getModelStatus').mockImplementation(() => status.promise);
  const generate = spyOn(apiClient, 'generateSpeech').mockResolvedValue({
    id: 'wrong-server',
  } as GenerationResponse);
  const form = await mountGeneration();
  let pending!: Promise<void>;
  act(() => {
    pending = form.generation.handleSubmit(
      { text: 'private original text', language: 'en', engine: 'qwen' },
      'original-profile',
    );
  });
  await act(async () => connect('generation-replacement'));
  expect(useServerStore.getState().connectionId).toBe('generation-replacement');
  await act(async () => {
    status.resolve({ models: [] });
    await pending;
  });
  expect(generate).not.toHaveBeenCalled();
  expect(form.onSuccess).not.toHaveBeenCalled();
  expect(useGenerationStore.getState().pendingGenerationIds.size).toBe(0);
});

test('generation on an unchanged connection retains request settings and tracks completion', async () => {
  spyOn(apiClient, 'getModelStatus').mockResolvedValue({ models: [] });
  const generate = spyOn(apiClient, 'generateSpeech').mockResolvedValue({
    id: 'new-generation',
  } as GenerationResponse);
  const form = await mountGeneration();
  await act(async () => {
    await form.generation.handleSubmit(
      { text: 'hello', language: 'en', engine: 'qwen', modelSize: '1.7B' },
      'profile',
    );
  });
  expect(generate).toHaveBeenCalledWith(
    expect.objectContaining({
      text: 'hello',
      profile_id: 'profile',
      model_size: '1.7B',
      engine: 'qwen',
      max_chunk_chars: 800,
    }),
  );
  expect(form.onSuccess).toHaveBeenCalledWith('new-generation');
  expect(useGenerationStore.getState().pendingGenerationIds.has('new-generation')).toBe(true);
});
