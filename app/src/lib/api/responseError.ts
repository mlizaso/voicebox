function formatErrorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string') return detail || fallback;
  if (Array.isArray(detail)) {
    return (
      detail
        .map((item) => formatErrorDetail(item, ''))
        .filter(Boolean)
        .join('; ') || fallback
    );
  }
  if (detail && typeof detail === 'object') {
    const error = detail as Record<string, unknown>;
    if (typeof error.msg === 'string') return error.msg || fallback;
    if (typeof error.message === 'string') return error.message || fallback;
    return JSON.stringify(detail);
  }
  return fallback;
}

/** Parse API failures consistently without consuming successful response streams. */
export async function assertApiResponse(response: Response): Promise<void> {
  if (response.ok) return;
  const fallback = `HTTP error! status: ${response.status}`;
  const body: unknown = await response
    .json()
    .catch(() => ({ detail: response.statusText || fallback }));
  const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : undefined;
  throw new Error(formatErrorDetail(detail, fallback));
}
