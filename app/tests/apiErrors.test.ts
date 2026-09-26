import { afterEach, expect, test } from 'bun:test';
import { apiClient } from '../src/lib/api/client';
import { useServerStore } from '../src/stores/serverStore';

const originalFetch = globalThis.fetch;
const originalConnection = useServerStore.getState();

afterEach(() => {
  globalThis.fetch = originalFetch;
  useServerStore.setState(originalConnection);
});

const requests = [
  { name: 'JSON', run: () => apiClient.listProfiles() },
  {
    name: 'multipart',
    run: () => apiClient.addProfileSample('voice', new File(['audio'], 'sample.wav'), 'hello'),
  },
  { name: 'export', run: () => apiClient.exportProfile('voice') },
  { name: 'blob', run: () => apiClient.previewEffects('generation', []) },
];

for (const request of requests) {
  test.each([
    { body: 'null', expected: 'HTTP error! status: 422' },
    { body: '{"detail":[null,{"msg":"Name required"}]}', expected: 'Name required' },
    { body: '{"detail":"Invalid voice"}', expected: 'Invalid voice' },
    { body: '{"detail":[{"msg":"First"},{"msg":"Second"}]}', expected: 'First; Second' },
    { body: '{"detail":{"message":"Unavailable"}}', expected: 'Unavailable' },
    { body: '{"detail":[]}', expected: 'HTTP error! status: 422' },
    { body: 'bad gateway', expected: 'Unprocessable Entity' },
  ])(`${request.name} preserves useful error details for $body`, async ({ body, expected }) => {
    useServerStore.setState({ serverUrl: 'http://127.0.0.1:17493', remoteApiToken: '' });
    globalThis.fetch = (async () =>
      new Response(body, {
        status: 422,
        statusText: 'Unprocessable Entity',
      })) as unknown as typeof fetch;
    await expect(request.run()).rejects.toThrow(expected);
  });
}

test('successful export response remains unread for bounded streaming consumers', async () => {
  useServerStore.setState({ serverUrl: 'http://127.0.0.1:17493', remoteApiToken: '' });
  const response = new Response('archive bytes');
  globalThis.fetch = (async () => response) as unknown as typeof fetch;
  const exported = await apiClient.exportProfile('voice');
  expect(exported).toBe(response);
  expect(exported.bodyUsed).toBe(false);
  expect(await exported.text()).toBe('archive bytes');
});
