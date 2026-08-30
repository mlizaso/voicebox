import { describe, expect, test } from 'bun:test';
import { bufferResponseBounded } from '../src/lib/api/boundedResponse';

describe('bufferResponseBounded', () => {
  test('returns a replayable response with the exact bytes and metadata', async () => {
    const response = new Response(new Uint8Array([1, 2, 3, 4]), {
      headers: { 'content-type': 'audio/wav' },
    });

    const buffered = await bufferResponseBounded(response, 4);

    expect(buffered.headers.get('content-length')).toBe('4');
    expect(buffered.headers.get('content-type')).toBe('audio/wav');
    expect([...new Uint8Array(await buffered.arrayBuffer())]).toEqual([1, 2, 3, 4]);
  });

  test('rejects an oversized declared length before reading', async () => {
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      cancel() {
        cancelled = true;
      },
    });
    const response = new Response(body, { headers: { 'content-length': '5' } });

    await expect(bufferResponseBounded(response, 4)).rejects.toThrow(
      'The exported file exceeds the allowed size',
    );
    expect(cancelled).toBe(true);
  });

  test('cancels a stream that crosses the runtime limit', async () => {
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array([1, 2, 3]));
        controller.enqueue(new Uint8Array([4, 5]));
      },
      cancel() {
        cancelled = true;
      },
    });

    await expect(bufferResponseBounded(new Response(body), 4)).rejects.toThrow(
      'The exported file exceeds the allowed size',
    );
    expect(cancelled).toBe(true);
  });
});
