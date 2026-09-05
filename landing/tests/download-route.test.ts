import { expect, test } from 'bun:test';
import { NextRequest } from 'next/server';
import { GET } from '../src/app/download/[platform]/route';

test.each([
  ['mac-arm', 'https://voicebox.sh/download?platform=macArm'],
  ['linux', 'https://voicebox.sh/linux-install'],
  ['unknown', 'https://voicebox.sh/download'],
])('download alias %s ignores untrusted forwarding headers', async (platform, location) => {
  const response = await GET(
    new NextRequest(`https://spoof.example/download/${platform}`, {
      headers: {
        host: 'spoof.example',
        'x-forwarded-host': 'spoof.example',
        'x-forwarded-proto': 'http',
      },
    }),
    { params: Promise.resolve({ platform }) },
  );
  expect(response.status).toBe(307);
  expect(response.headers.get('location')).toBe(location);
});
