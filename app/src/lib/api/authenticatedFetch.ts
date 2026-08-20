import { useServerStore } from '@/stores/serverStore';

function isLoopbackHostname(hostname: string): boolean {
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
  if (voiceboxRequest && !isSecureVoiceboxServerUrl(state.serverUrl)) {
    return Promise.reject(
      new TypeError('Refusing to send a Voicebox request to a non-loopback server over HTTP'),
    );
  }
  if (state.remoteApiToken && voiceboxRequest && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${state.remoteApiToken}`);
  }
  return fetch(input, {
    ...init,
    credentials:
      init.credentials ??
      (voiceboxRequest
        ? 'include'
        : ((input instanceof Request ? input.credentials : undefined) ?? 'same-origin')),
    headers,
  });
}

/** Open a Voicebox event stream with its HttpOnly remote session cookie. */
export function authenticatedEventSource(input: string | URL): EventSource {
  const state = useServerStore.getState();
  const voiceboxRequest = isVoiceboxRequest(input);
  if (voiceboxRequest && !isSecureVoiceboxServerUrl(state.serverUrl)) {
    throw new TypeError(
      'Refusing to open a Voicebox event stream to a non-loopback server over HTTP',
    );
  }
  return new EventSource(input, { withCredentials: voiceboxRequest });
}
