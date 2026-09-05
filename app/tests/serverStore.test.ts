import { beforeEach, expect, test } from 'bun:test';
import { QueryObserver } from '@tanstack/react-query';

const values = new Map<string, string>();
globalThis.localStorage = {
  getItem: (key) => values.get(key) ?? null,
  setItem: (key, value) => {
    values.set(key, value);
  },
  removeItem: (key) => {
    values.delete(key);
  },
  clear: () => values.clear(),
  key: (index) => [...values.keys()][index] ?? null,
  get length() {
    return values.size;
  },
};

const { useServerStore } = await import('../src/stores/serverStore');
const { queryClient } = await import('../src/lib/queryClient');
const { usePlayerStore } = await import('../src/stores/playerStore');
const sharedConnection = {
  connectionId: 'main-window-connection',
  serverUrl: 'https://b.example',
  remoteApiToken: 'session-only-secret',
  mode: 'remote' as const,
};

beforeEach(() => {
  queryClient.clear();
  useServerStore.setState({
    connectionId: 'initial',
    serverUrl: 'https://a.example',
    remoteApiToken: '',
    mode: 'local',
  });
});

test.each([
  'url',
  'token',
  'mode',
  'snapshot',
])('changing %s clears old data even when the new server fails', async (field) => {
  queryClient.setQueryData(['profiles'], [{ id: 'private-a' }]);
  usePlayerStore.getState().setAudio('https://a.example/private.wav', 'private-a', null);
  const observer = new QueryObserver(queryClient, {
    queryKey: ['profiles'],
    queryFn: async () => {
      throw new Error('Unauthorized');
    },
    retry: false,
  });
  const unsubscribe = observer.subscribe(() => {});
  if (field === 'url') useServerStore.getState().setServerUrl('https://b.example');
  if (field === 'token') useServerStore.getState().setRemoteApiToken('different-account');
  if (field === 'mode') useServerStore.getState().setMode('remote');
  if (field === 'snapshot') useServerStore.getState().setConnection(sharedConnection);
  expect(observer.getCurrentResult().data).toBeUndefined();
  expect(usePlayerStore.getState().audioUrl).toBeNull();
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(observer.getCurrentResult().data).toBeUndefined();
  unsubscribe();
});

test('removes legacy persisted credentials and persists only preferences', async () => {
  values.set(
    'voicebox-server',
    JSON.stringify({
      version: 0,
      state: {
        serverUrl: 'https://legacy.example',
        remoteApiToken: 'legacy-secret',
        mode: 'remote',
      },
    }),
  );
  await useServerStore.persist.rehydrate();
  expect(useServerStore.getState().serverUrl).toBe('https://legacy.example');
  expect(useServerStore.getState().remoteApiToken).toBe('');
  expect(values.get('voicebox-server')).not.toContain('legacy-secret');
  useServerStore.getState().setRemoteApiToken('session-secret');
  expect(useServerStore.getState().remoteApiToken).toBe('session-secret');
  expect(values.get('voicebox-server')).not.toContain('session-secret');
});

test('shares a connection atomically without persisting its token or identity', () => {
  const observed: string[][] = [];
  const unsubscribe = useServerStore.subscribe((state) => {
    observed.push([state.connectionId, state.serverUrl, state.remoteApiToken, state.mode]);
  });
  try {
    expect(useServerStore.getState().setConnection(sharedConnection)).toBe(true);
    expect(observed).toEqual([
      ['main-window-connection', 'https://b.example', 'session-only-secret', 'remote'],
    ]);
    const persisted = values.get('voicebox-server');
    expect(persisted).not.toContain('session-only-secret');
    expect(persisted).not.toContain('main-window-connection');
  } finally {
    unsubscribe();
  }
});

test('returning to the same server cannot accept events from its earlier connection', () => {
  const oldId = useServerStore.getState().connectionId;
  useServerStore.getState().setServerUrl('https://b.example');
  useServerStore.getState().setServerUrl('https://a.example');
  const newId = useServerStore.getState().connectionId;
  expect(newId).not.toBe(oldId);
  useServerStore.getState().setServerUrl('https://a.example');
  expect(useServerStore.getState().connectionId).toBe(newId);
});

test('late requests from the old server cannot repopulate its cache', async () => {
  let resolveOld: (value: string) => void = () => {};
  const pending = queryClient
    .fetchQuery({
      queryKey: ['history'],
      queryFn: () =>
        new Promise<string>((resolve) => {
          resolveOld = resolve;
        }),
    })
    .catch(() => undefined);
  useServerStore.getState().setServerUrl('https://b.example');
  resolveOld('private-a');
  await pending;
  expect(queryClient.getQueryData(['history'])).toBeUndefined();
});

test.each([
  'url',
  'token',
  'mode',
  'snapshot',
])('changing %s waits for pending writes and their callbacks', async (field) => {
  let finish: () => void = () => {};
  let entered: () => void = () => {};
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const mutation = queryClient.getMutationCache().build(queryClient, {
    mutationFn: () =>
      new Promise<string>((resolve) => {
        finish = () => resolve('private-a');
        entered();
      }),
    onSuccess: (data) => {
      queryClient.setQueryData(['capture'], data);
    },
  });
  const pending = mutation.execute(undefined);
  await started;
  const change = () => {
    if (field === 'url') useServerStore.getState().setServerUrl('https://b.example');
    if (field === 'token') useServerStore.getState().setRemoteApiToken('different-account');
    if (field === 'mode') useServerStore.getState().setMode('remote');
    if (field === 'snapshot') useServerStore.getState().setConnection(sharedConnection);
  };
  change();
  expect(useServerStore.getState().serverUrl).toBe('https://a.example');
  expect(useServerStore.getState().remoteApiToken).toBe('');
  expect(useServerStore.getState().mode).toBe('local');
  finish();
  await pending;
  expect(queryClient.getQueryData<string>(['capture'])).toBe('private-a');
  change();
  expect(queryClient.getQueryData(['capture'])).toBeUndefined();
});
