import { EventSource as FetchEventSource } from 'eventsource';
import { useServerStore } from '@/stores/serverStore';

export function isLoopbackHostname(hostname: string): boolean {
  const normalized = hostname.toLowerCase().replace(/\.$/, '');
  if (
    normalized === 'localhost' ||
    normalized === 'tauri.localhost' ||
    normalized === '[::1]' ||
    normalized === '::1'
  ) {
    return true;
  }
  const ipv4Parts = normalized.split('.');
  return (
    ipv4Parts.length === 4 &&
    ipv4Parts[0] === '127' &&
    ipv4Parts.every((part) => /^\d{1,3}$/.test(part) && Number(part) <= 255)
  );
}

/** Whether a server URL can carry a bearer/cookie capability without plaintext exposure. */
export function isSecureVoiceboxServerUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return (
      url.protocol === 'https:' || (url.protocol === 'http:' && isLoopbackHostname(url.hostname))
    );
  } catch {
    return false;
  }
}

/** Remote sessions require the explicitly configured capability, even if a
 * browser retained an HttpOnly cookie from an earlier login. */
export function validateVoiceboxConnection(): void {
  const state = useServerStore.getState();
  if (!isSecureVoiceboxServerUrl(state.serverUrl)) {
    throw new TypeError('Refusing to send a Voicebox request to a non-loopback server over HTTP');
  }
  if (!isLoopbackHostname(new URL(state.serverUrl).hostname) && !state.remoteApiToken) {
    throw new TypeError('Enter the remote API token to connect to this server');
  }
}

function isVoiceboxRequest(input: RequestInfo | URL): boolean {
  try {
    const requestUrl = new URL(
      input instanceof Request ? input.url : input.toString(),
      window.location.origin,
    );
    const serverUrl = new URL(useServerStore.getState().serverUrl);
    return requestUrl.origin === serverUrl.origin;
  } catch {
    return false;
  }
}

/** Fetch a Voicebox resource with the configured remote bearer capability. */
export function authenticatedFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(
    init.headers ?? (input instanceof Request ? input.headers : undefined),
  );
  const state = useServerStore.getState();
  const voiceboxRequest = isVoiceboxRequest(input);
  if (voiceboxRequest) {
    try {
      validateVoiceboxConnection();
    } catch (error) {
      return Promise.reject(error);
    }
  }
  if (state.remoteApiToken && voiceboxRequest && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${state.remoteApiToken}`);
  }
  return fetch(input, {
    ...init,
    // Every remote request uses the configured bearer, including event streams.
    // Never silently reuse a previous account's HttpOnly session cookie.
    credentials: voiceboxRequest
      ? 'omit'
      : (init.credentials ?? (input instanceof Request ? input.credentials : 'same-origin')),
    headers,
  });
}

/** Keep the standard EventSource API while authenticating every reconnect. */
export function authenticatedEventSource(input: string | URL) {
  const state = useServerStore.getState();
  const source = new FetchEventSource(input, {
    fetch: authenticatedFetch,
    maxBufferSize: 1024 * 1024,
  });
  const unsubscribe = useServerStore.subscribe((current) => {
    if (
      current.serverUrl !== state.serverUrl ||
      current.remoteApiToken !== state.remoteApiToken ||
      current.mode !== state.mode
    ) {
      source.close();
    }
  });
  const close = source.close.bind(source);
  source.close = () => {
    unsubscribe();
    close();
  };
  return source;
}
