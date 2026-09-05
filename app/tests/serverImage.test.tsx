import { afterEach, expect, mock, test } from 'bun:test';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import { ServerImage } from '../src/components/ServerImage';
import { useServerStore } from '../src/stores/serverStore';

const originalFetch = globalThis.fetch;
const originalWindow = globalThis.window;
let renderer: ReactTestRenderer | undefined;

afterEach(() => {
  act(() => renderer?.unmount());
  renderer = undefined;
  globalThis.fetch = originalFetch;
  globalThis.window = originalWindow;
  useServerStore.setState({ serverUrl: 'http://127.0.0.1:17493', remoteApiToken: '' });
});

test('private avatars authenticate, release their blobs, and reset on a connection switch', async () => {
  globalThis.window = { location: { origin: 'tauri://localhost' } } as Window & typeof globalThis;
  useServerStore.setState({
    connectionId: 'image-first',
    serverUrl: 'https://remote.example',
    remoteApiToken: 'private-capability',
  });
  let requestSignal: AbortSignal | null | undefined;
  const request = mock(async (_input: RequestInfo | URL, init?: RequestInit) => {
    expect(new Headers(init?.headers).get('Authorization')).toBe('Bearer private-capability');
    expect(init?.credentials).toBe('omit');
    requestSignal = init?.signal;
    return new Response('avatar bytes', { headers: { 'content-type': 'image/png' } });
  });
  globalThis.fetch = request as unknown as typeof fetch;
  await act(async () => {
    renderer = create(
      <ServerImage
        src="https://remote.example/profiles/id/avatar"
        alt="Narrator"
        fallback={<span>missing</span>}
      />,
    );
  });
  const url: string = renderer!.root.findByType('img').props.src;
  expect(url.startsWith('blob:')).toBe(true);
  expect(await (await originalFetch(url)).text()).toBe('avatar bytes');
  expect(request).toHaveBeenCalledTimes(1);
  await act(async () => {
    useServerStore.setState({ connectionId: 'image-second', serverUrl: 'https://other.example' });
  });
  expect(renderer!.root.findByType('span').children).toEqual(['missing']);
  expect(requestSignal?.aborted).toBe(true);
  await expect(originalFetch(url)).rejects.toThrow();
  expect(request).toHaveBeenCalledTimes(1);
});

test('upload previews need no network and decode failures retain the placeholder', () => {
  const request = mock(() => Promise.reject(new Error('Unexpected request')));
  globalThis.fetch = request as unknown as typeof fetch;
  act(() => {
    renderer = create(
      <ServerImage
        src="data:image/png;base64,AAAA"
        alt="Preview"
        fallback={<span>missing</span>}
      />,
    );
  });
  act(() => renderer!.root.findByType('img').props.onError());
  expect(renderer!.root.findByType('span').children).toEqual(['missing']);
  expect(request).not.toHaveBeenCalled();
});
