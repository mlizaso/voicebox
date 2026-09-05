import { useServerStore } from '@/stores/serverStore';
import {
  authenticatedFetch,
  isLoopbackHostname,
  validateVoiceboxConnection,
} from './authenticatedFetch';
import { bufferResponseBounded } from './boundedResponse';

/** Local clips can stream directly. Remote clips need a bearer header, which
 * media elements cannot send; use a bounded, revocable object URL there. */
export async function loadAudioSource(url: string, signal: AbortSignal): Promise<string> {
  validateVoiceboxConnection();
  if (new URL(url).origin !== new URL(useServerStore.getState().serverUrl).origin) {
    throw new Error('Audio does not belong to the connected server');
  }
  if (isLoopbackHostname(new URL(url).hostname) && !useServerStore.getState().remoteApiToken) {
    signal.throwIfAborted();
    return url;
  }
  const response = await authenticatedFetch(url, { signal });
  if (!response.ok) throw new Error(`Could not load audio (${response.status})`);
  const bounded = await bufferResponseBounded(response, 32 * 1024 * 1024);
  const blob = await bounded.blob();
  signal.throwIfAborted();
  return URL.createObjectURL(blob);
}

export function releaseAudioSource(url: string): void {
  if (url.startsWith('blob:')) URL.revokeObjectURL(url);
}
