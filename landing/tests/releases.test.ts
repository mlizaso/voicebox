import { afterEach, expect, test } from 'bun:test';
import { getLatestRelease } from '../src/lib/releases';

const originalFetch = globalThis.fetch;
const originalNow = Date.now;
afterEach(() => {
  globalThis.fetch = originalFetch;
  Date.now = originalNow;
});

test('concurrent release reads share a refresh and failed pagination preserves the full count', async () => {
  let now = originalNow();
  Date.now = () => now;
  let calls = 0;
  let failSecondPage = false;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls++;
    expect(init?.signal).toBeInstanceOf(AbortSignal);
    if (String(input).endsWith('/latest')) {
      return Response.json({ tag_name: 'v1.0.0', assets: [] });
    }
    if (new URL(String(input)).searchParams.get('page') === '1') {
      return Response.json(
        Array.from({ length: 100 }, () => ({ assets: [{ download_count: 2 }] })),
      );
    }
    if (failSecondPage) return new Response('rate limited', { status: 429 });
    return Response.json([{ assets: [{ download_count: 3 }] }]);
  }) as typeof fetch;
  const results = await Promise.all(Array.from({ length: 20 }, () => getLatestRelease()));
  expect(calls).toBe(3);
  expect(results.every((release) => release.totalDownloads === 203)).toBe(true);
  now += 6 * 60 * 1000;
  failSecondPage = true;
  const stale = await getLatestRelease();
  expect(stale.totalDownloads).toBe(203);
  expect(calls).toBe(6);
  failSecondPage = false;
  expect((await getLatestRelease()).totalDownloads).toBe(203);
  expect(calls).toBe(9); // A partial count was never marked as a successful refresh.
});
