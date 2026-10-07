import { afterEach, beforeEach, expect, mock, spyOn, test } from 'bun:test';
import { defaultScheduler, notifyManager, QueryClientProvider } from '@tanstack/react-query';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import * as auth from '../src/lib/api/authenticatedFetch';
import { apiClient } from '../src/lib/api/client';
import type {
  ActiveDownloadTask,
  ActiveTasksResponse,
  StoryItemDetail,
} from '../src/lib/api/types';
import { useGenerationProgress } from '../src/lib/hooks/useGenerationProgress';
import { useRestoreActiveTasks } from '../src/lib/hooks/useRestoreActiveTasks';
import { queryClient } from '../src/lib/queryClient';
import { useGenerationStore } from '../src/stores/generationStore';
import { usePlayerStore } from '../src/stores/playerStore';
import { useServerStore } from '../src/stores/serverStore';

let renderer: ReactTestRenderer | undefined;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function tasks(id: string): ActiveTasksResponse {
  return {
    downloads: [{ model_name: id, status: 'downloading', started_at: '2026-01-01' }],
    generations: [
      { task_id: id, profile_id: 'profile', text_preview: id, started_at: '2026-01-01' },
    ],
  };
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
  // Flush query notifications within async act rather than an unrelated timer turn.
  notifyManager.setScheduler(queueMicrotask);
  queryClient.clear();
  connect('tasks-first');
  useGenerationStore.setState({
    pendingGenerationIds: new Set(),
    pendingStoryAdds: new Map(),
    isGenerating: false,
    activeGenerationId: null,
  });
  usePlayerStore.getState().reset();
});

afterEach(() => {
  act(() => renderer?.unmount());
  renderer = undefined;
  mock.restore();
  queryClient.clear();
  notifyManager.setScheduler(defaultScheduler);
  useGenerationStore.setState({
    pendingGenerationIds: new Set(),
    pendingStoryAdds: new Map(),
    isGenerating: false,
    activeGenerationId: null,
  });
  usePlayerStore.getState().reset();
});

test('task polling ignores the previous connection and immediately restores the new one', async () => {
  const oldTasks = deferred<ActiveTasksResponse>();
  const newTasks = deferred<ActiveTasksResponse>();
  const request = spyOn(apiClient, 'getActiveTasks')
    .mockImplementationOnce(() => oldTasks.promise)
    .mockImplementationOnce(() => newTasks.promise);
  let downloads: ActiveDownloadTask[] = [];
  function Restore() {
    downloads = useRestoreActiveTasks();
    return null;
  }
  act(() => {
    renderer = create(<Restore />);
  });
  await act(async () => connect('tasks-second'));
  await act(async () => oldTasks.resolve(tasks('old-task')));
  expect(useGenerationStore.getState().pendingGenerationIds.size).toBe(0);
  expect(useGenerationStore.getState().activeGenerationId).toBeNull();
  expect(downloads).toEqual([]);
  expect(request).toHaveBeenCalledTimes(2);
  await act(async () => newTasks.resolve(tasks('new-task')));
  expect([...useGenerationStore.getState().pendingGenerationIds]).toEqual(['new-task']);
  expect(useGenerationStore.getState().activeGenerationId).toBe('new-task');
  expect(downloads.map((download) => download.model_name)).toEqual(['new-task']);
});

test('a connection switch immediately hides previously restored downloads', async () => {
  const nextTasks = deferred<ActiveTasksResponse>();
  spyOn(apiClient, 'getActiveTasks')
    .mockResolvedValueOnce(tasks('old-download'))
    .mockImplementationOnce(() => nextTasks.promise);
  let downloads: ActiveDownloadTask[] = [];
  function Restore() {
    downloads = useRestoreActiveTasks();
    return null;
  }
  await act(async () => {
    renderer = create(<Restore />);
  });
  expect(downloads).toHaveLength(1);
  await act(async () => connect('tasks-second'));
  expect(downloads).toEqual([]);
  await act(async () => nextTasks.resolve({ downloads: [], generations: [] }));
});

test('an unmounted polling hook cannot restore generations from a late response', async () => {
  const response = deferred<ActiveTasksResponse>();
  spyOn(apiClient, 'getActiveTasks').mockImplementation(() => response.promise);
  function Restore() {
    useRestoreActiveTasks();
    return null;
  }
  act(() => {
    renderer = create(<Restore />);
  });
  act(() => {
    renderer?.unmount();
    renderer = undefined;
  });
  await act(async () => response.resolve(tasks('late-task')));
  expect(useGenerationStore.getState().pendingGenerationIds.size).toBe(0);
  expect(useGenerationStore.getState().activeGenerationId).toBeNull();
});

test('polling does not overlap requests or restart from a queued timer after unmount', async () => {
  const first = deferred<ActiveTasksResponse>();
  const request = spyOn(apiClient, 'getActiveTasks')
    .mockImplementationOnce(() => first.promise)
    .mockResolvedValue({ downloads: [], generations: [] });
  const originalSetInterval = globalThis.setInterval;
  let poll!: () => Promise<void>;
  spyOn(globalThis, 'setInterval').mockImplementation(((callback: () => void, delay?: number) => {
    poll = callback as () => Promise<void>;
    return originalSetInterval(callback, delay);
  }) as typeof setInterval);
  function Restore() {
    useRestoreActiveTasks();
    return null;
  }
  act(() => {
    renderer = create(<Restore />);
  });
  await act(async () => poll());
  expect(request).toHaveBeenCalledTimes(1);
  await act(async () => first.resolve(tasks('generation')));
  await act(async () => poll());
  expect(request).toHaveBeenCalledTimes(2);
  expect(useGenerationStore.getState().activeGenerationId).toBeNull();
  act(() => {
    renderer?.unmount();
    renderer = undefined;
  });
  await poll();
  expect(request).toHaveBeenCalledTimes(2);
});

class FakeEventSource {
  CLOSED = 2;
  readyState = 0;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  close = mock(() => {});
}

async function mountProgress() {
  const sources: FakeEventSource[] = [];
  spyOn(auth, 'authenticatedEventSource').mockImplementation(() => {
    const source = new FakeEventSource();
    sources.push(source);
    return source as unknown as ReturnType<typeof auth.authenticatedEventSource>;
  });
  spyOn(apiClient, 'getGenerationSettings').mockResolvedValue({
    autoplay_on_generate: true,
    crossfade_ms: 0,
    max_chunk_chars: 100,
    normalize_audio: false,
  });
  useGenerationStore.getState().addPendingGeneration('generation');
  useGenerationStore.getState().addPendingStoryAdd('generation', 'old-story');
  function Progress() {
    useGenerationProgress();
    return null;
  }
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={queryClient}>
        <Progress />
      </QueryClientProvider>,
    );
  });
  return sources;
}

test.each([
  'completed',
  'failed',
  'not_found',
  'error',
])('queued %s events from a closed connection cannot affect the new connection', async (status) => {
  const sources = await mountProgress();
  const oldSource = sources[0];
  const oldMessage = oldSource.onmessage!;
  const oldError = oldSource.onerror!;
  const addStory = spyOn(apiClient, 'addStoryItem').mockResolvedValue({} as StoryItemDetail);
  const refetch = spyOn(queryClient, 'refetchQueries');
  await act(async () => {
    connect('tasks-second');
    // The same server may use the same generation IDs after reconnecting.
    useGenerationStore.getState().addPendingGeneration('generation');
    useGenerationStore.getState().addPendingStoryAdd('generation', 'new-story');
  });
  refetch.mockClear();
  await act(async () => {
    if (status === 'error') oldError();
    else oldMessage({ data: JSON.stringify({ id: 'generation', status }) });
  });
  expect(addStory).not.toHaveBeenCalled();
  expect(refetch).not.toHaveBeenCalled();
  expect(useGenerationStore.getState().pendingGenerationIds.has('generation')).toBe(true);
  expect(useGenerationStore.getState().pendingStoryAdds.get('generation')).toBe('new-story');
  expect(usePlayerStore.getState().audioId).toBeNull();
  expect(oldSource.close).toHaveBeenCalled();
  expect(sources).toHaveLength(2);
});

test('current generation completion adds and autoplays exactly once', async () => {
  const sources = await mountProgress();
  const message = sources[0].onmessage!;
  const addStory = spyOn(apiClient, 'addStoryItem').mockResolvedValue({} as StoryItemDetail);
  const refetch = spyOn(queryClient, 'refetchQueries');
  await act(async () => {
    const event = { data: JSON.stringify({ id: 'generation', status: 'completed', duration: 1 }) };
    message(event);
    message(event);
  });
  expect(addStory).toHaveBeenCalledTimes(1);
  expect(addStory).toHaveBeenCalledWith('old-story', { generation_id: 'generation' });
  expect(
    refetch.mock.calls.filter(([filters]) => filters?.queryKey?.[0] === 'history'),
  ).toHaveLength(1);
  expect(useGenerationStore.getState().pendingGenerationIds.size).toBe(0);
  expect(useGenerationStore.getState().pendingStoryAdds.size).toBe(0);
  expect(usePlayerStore.getState().audioId).toBe('generation');
  expect(usePlayerStore.getState().shouldAutoPlay).toBe(true);
});

test('a transient stream error retains the pending story add until terminal status reconnects', async () => {
  const sources = await mountProgress();
  const source = sources[0];
  const addStory = spyOn(apiClient, 'addStoryItem').mockResolvedValue({} as StoryItemDetail);
  await act(async () => source.onerror!());
  expect(source.close).not.toHaveBeenCalled();
  expect(useGenerationStore.getState().pendingGenerationIds.has('generation')).toBe(true);
  expect(useGenerationStore.getState().pendingStoryAdds.get('generation')).toBe('old-story');
  await act(async () =>
    source.onmessage!({
      data: JSON.stringify({ id: 'generation', status: 'completed', duration: 1 }),
    }),
  );
  expect(addStory).toHaveBeenCalledTimes(1);
  expect(usePlayerStore.getState().audioId).toBe('generation');
  expect(useGenerationStore.getState().pendingGenerationIds.size).toBe(0);
  expect(useGenerationStore.getState().pendingStoryAdds.size).toBe(0);
});

test('a terminal stream rejection clears tracking and its deferred story intent', async () => {
  const sources = await mountProgress();
  const source = sources[0];
  source.readyState = source.CLOSED;
  await act(async () => source.onerror!());
  expect(source.close).toHaveBeenCalledTimes(1);
  expect(useGenerationStore.getState().pendingGenerationIds.size).toBe(0);
  expect(useGenerationStore.getState().pendingStoryAdds.size).toBe(0);
  expect(usePlayerStore.getState().audioId).toBeNull();
});

test.each([
  'disconnect',
  '403 rejection',
])('the real EventSource handles %s without an active-task poll', async (failure) => {
  let attempts = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => {
    attempts += 1;
    if (failure === '403 rejection') return new Response('Forbidden', { status: 403 });
    return new Response(
      attempts === 1
        ? 'retry: 1\n\n'
        : 'data: {"id":"generation","status":"completed","duration":1}\n\n',
      { headers: { 'content-type': 'text/event-stream' } },
    );
  }) as unknown as typeof fetch;
  const addStory = spyOn(apiClient, 'addStoryItem').mockResolvedValue({} as StoryItemDetail);
  spyOn(apiClient, 'getGenerationSettings').mockResolvedValue({
    autoplay_on_generate: true,
    crossfade_ms: 0,
    max_chunk_chars: 100,
    normalize_audio: false,
  });
  useGenerationStore.getState().addPendingGeneration('generation');
  useGenerationStore.getState().addPendingStoryAdd('generation', 'story');
  let complete!: () => void;
  const completion = new Promise<void>((resolve) => {
    complete = resolve;
  });
  const unsubscribe = useGenerationStore.subscribe((state) => {
    if (!state.pendingGenerationIds.has('generation')) complete();
  });
  function Progress() {
    useGenerationProgress();
    return null;
  }
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    act(() => {
      renderer = create(
        <QueryClientProvider client={queryClient}>
          <Progress />
        </QueryClientProvider>,
      );
    });
    await act(async () => {
      await Promise.race([
        completion,
        new Promise<never>((_resolve, reject) => {
          timeout = setTimeout(
            () => reject(new Error('Completion was lost after disconnect')),
            1000,
          );
        }),
      ]);
    });
    expect(attempts).toBe(failure === 'disconnect' ? 2 : 1);
    if (failure === 'disconnect') {
      expect(addStory).toHaveBeenCalledWith('story', { generation_id: 'generation' });
      expect(usePlayerStore.getState().audioId).toBe('generation');
    } else {
      expect(addStory).not.toHaveBeenCalled();
      expect(usePlayerStore.getState().audioId).toBeNull();
    }
    expect(useGenerationStore.getState().pendingGenerationIds.size).toBe(0);
    expect(useGenerationStore.getState().pendingStoryAdds.size).toBe(0);
  } finally {
    unsubscribe();
    clearTimeout(timeout);
    act(() => {
      renderer?.unmount();
      renderer = undefined;
    });
    globalThis.fetch = originalFetch;
  }
});

test.each([
  'unmounted',
  'replaced',
])('callbacks from a subscription that was %s are ignored without a connection change', async (end) => {
  const sources = await mountProgress();
  const message = sources[0].onmessage!;
  const addStory = spyOn(apiClient, 'addStoryItem').mockResolvedValue({} as StoryItemDetail);
  act(() => {
    if (end === 'unmounted') {
      renderer?.unmount();
      renderer = undefined;
    } else {
      useGenerationStore.getState().removePendingGeneration('generation');
    }
  });
  if (end === 'replaced') {
    act(() => useGenerationStore.getState().addPendingGeneration('generation'));
    expect(sources).toHaveLength(2);
  }
  await act(async () => {
    message({ data: JSON.stringify({ id: 'generation', status: 'completed' }) });
  });
  expect(addStory).not.toHaveBeenCalled();
  expect(useGenerationStore.getState().pendingGenerationIds.has('generation')).toBe(true);
  expect(usePlayerStore.getState().audioId).toBeNull();
});

test.each([
  'connection switch',
  'unmount',
])('an old story-add result cannot invalidate data after %s', async (end) => {
  const sources = await mountProgress();
  const result = deferred<StoryItemDetail>();
  spyOn(apiClient, 'addStoryItem').mockImplementation(() => result.promise);
  const invalidate = spyOn(queryClient, 'invalidateQueries');
  await act(async () => {
    sources[0].onmessage!({
      data: JSON.stringify({ id: 'generation', status: 'completed' }),
    });
  });
  await act(async () => {
    if (end === 'connection switch') connect('tasks-second');
    else {
      renderer?.unmount();
      renderer = undefined;
    }
  });
  await act(async () => result.resolve({} as StoryItemDetail));
  expect(invalidate).not.toHaveBeenCalled();
});
