/** Materialize a response body without ever accepting more than ``maxBytes``. */
export async function bufferResponseBounded(
  response: Response,
  maxBytes: number,
): Promise<Response> {
  if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
    throw new Error('Invalid download size limit');
  }

  const rawLength = response.headers.get('content-length');
  const declaredLength = rawLength === null ? null : Number(rawLength);
  if (
    declaredLength !== null &&
    Number.isSafeInteger(declaredLength) &&
    declaredLength > maxBytes
  ) {
    await response.body?.cancel();
    throw new Error('The exported file exceeds the allowed size');
  }

  const body = response.body;
  if (!body) {
    throw new Error('The export response did not contain a downloadable body');
  }

  const reader = body.getReader();
  const chunks: ArrayBuffer[] = [];
  let receivedBytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      receivedBytes += value.byteLength;
      if (receivedBytes > maxBytes) {
        throw new Error('The exported file exceeds the allowed size');
      }
      const chunk = new ArrayBuffer(value.byteLength);
      new Uint8Array(chunk).set(value);
      chunks.push(chunk);
    }
  } catch (error) {
    await reader.cancel(error).catch(() => undefined);
    throw error;
  } finally {
    reader.releaseLock();
  }

  const contentType = response.headers.get('content-type') || 'application/octet-stream';
  return new Response(new Blob(chunks, { type: contentType }), {
    headers: {
      'content-length': String(receivedBytes),
      'content-type': contentType,
    },
  });
}
