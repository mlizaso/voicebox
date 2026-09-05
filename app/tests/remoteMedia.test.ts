import { afterEach, expect, mock, test } from 'bun:test';
import { QueryObserver } from '@tanstack/react-query';
import { loadAudioSource, releaseAudioSource } from '../src/lib/api/audioSource';
import { authenticatedEventSource, authenticatedFetch } from '../src/lib/api/authenticatedFetch';
import { queryClient } from '../src/lib/queryClient';
import { useServerStore } from '../src/stores/serverStore';

const originalFetch = globalThis.fetch;
const originalWindow = globalThis.window;

function connect(url = 'https://remote.example', token = 'test-capability') {
  globalThis.window = { location: { origin: 'tauri://localhost' } } as Window & typeof globalThis;
  useServerStore.setState({ serverUrl: url, remoteApiToken: token });
}

afterEach(() => {
  queryClient.clear();
  globalThis.fetch = originalFetch;
  globalThis.window = originalWindow;
  useServerStore.setState({ serverUrl: 'http://127.0.0.1:17493', remoteApiToken: '' });
});

test('switching servers clears credentials before active queries refetch', async () => {
  connect();
  queryClient.clear();
  const request = mock(async () => new Response('{}'));
  globalThis.fetch = request as unknown as typeof fetch;
  const observer = new QueryObserver(queryClient, {
    queryKey: ['health'],
    queryFn: () => authenticatedFetch(`${useServerStore.getState().serverUrl}/health`),
    retry: false,
  });
  const unsubscribe = observer.subscribe(() => {});
  try {
    await observer.refetch();
    expect(request).toHaveBeenCalledTimes(1);
    request.mockClear();
    useServerStore.getState().setServerUrl('https://new.example');
    const result = await observer.refetch();
    expect(request).not.toHaveBeenCalled();
    expect(useServerStore.getState().remoteApiToken).toBe('');
    expect(result.error?.message).toContain('token');
  } finally {
    unsubscribe();
  }
});

test('remote media sends its bearer header and does not use cookies', async () => {
  connect();
  const request = mock(async (_input: RequestInfo | URL, init?: RequestInit) => {
    expect(new Headers(init?.headers).get('Authorization')).toBe('Bearer test-capability');
    expect(init?.credentials).toBe('omit');
    return new Response('synthetic audio', { headers: { 'content-type': 'audio/wav' } });
  });
  globalThis.fetch = request as unknown as typeof fetch;
  const url = await loadAudioSource(
    'https://remote.example/audio/id',
    new AbortController().signal,
  );
  expect(url.startsWith('blob:')).toBe(true);
  expect(request).toHaveBeenCalledTimes(1);
  releaseAudioSource(url);
});

test('cleared tokens and insecure remote URLs fail before sending media or API requests', async () => {
  const request = mock(async () => new Response('private cookie-authenticated data'));
  globalThis.fetch = request as unknown as typeof fetch;
  connect('https://remote.example', '');
  await expect(authenticatedFetch('https://remote.example/profiles')).rejects.toThrow('token');
  await expect(
    loadAudioSource('https://remote.example/audio/id', new AbortController().signal),
  ).rejects.toThrow('token');
  connect('http://remote.example');
  await expect(
    loadAudioSource('http://remote.example/audio/id', new AbortController().signal),
  ).rejects.toThrow('HTTP');
  expect(request).not.toHaveBeenCalled();
});

test('remote media rejects oversized responses and cancellation', async () => {
  connect();
  globalThis.fetch = (async () =>
    new Response('small body', {
      headers: { 'content-length': String(33 * 1024 * 1024) },
    })) as unknown as typeof fetch;
  await expect(
    loadAudioSource('https://remote.example/audio/id', new AbortController().signal),
  ).rejects.toThrow('allowed size');
  connect('http://127.0.0.1:17493', '');
  const controller = new AbortController();
  controller.abort();
  await expect(
    loadAudioSource('http://127.0.0.1:17493/audio/id', controller.signal),
  ).rejects.toThrow();
});

test('event streams authenticate without cookies and close when the connection changes', async () => {
  connect();
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({
    start(value) {
      controller = value;
    },
  });
  const request = mock(async (_input: RequestInfo | URL, init?: RequestInit) => {
    expect(new Headers(init?.headers).get('Authorization')).toBe('Bearer test-capability');
    expect(init?.credentials).toBe('omit');
    return new Response(body, { headers: { 'content-type': 'text/event-stream' } });
  });
  globalThis.fetch = request as unknown as typeof fetch;
  const source = authenticatedEventSource('https://remote.example/events');
  try {
    const message = new Promise<string>((resolve, reject) => {
      source.onmessage = (event) => resolve(event.data);
      source.onerror = () => reject(new Error('Stream failed'));
    });
    const encoded = new TextEncoder().encode('data: café\r\n\r\n');
    controller.enqueue(encoded.slice(0, 10));
    controller.enqueue(encoded.slice(10));
    expect(await message).toBe('café');
    useServerStore.getState().setRemoteApiToken('');
    expect(source.readyState).toBe(source.CLOSED);
    expect(request).toHaveBeenCalledTimes(1);
  } finally {
    source.close();
  }
});
