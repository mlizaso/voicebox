import { type ImgHTMLAttributes, type ReactNode, useEffect, useState } from 'react';
import { authenticatedFetch, validateVoiceboxConnection } from '@/lib/api/authenticatedFetch';
import { bufferResponseBounded } from '@/lib/api/boundedResponse';
import { useServerStore } from '@/stores/serverStore';

type ServerImageProps = Omit<ImgHTMLAttributes<HTMLImageElement>, 'src' | 'srcSet' | 'onError'> & {
  src: string;
  fallback: ReactNode;
};

/** Private avatars require bearer headers; local upload previews remain local. */
export function ServerImage({ src, fallback, alt, ...props }: ServerImageProps) {
  const connectionId = useServerStore((state) => state.connectionId);
  const [loaded, setLoaded] = useState<{ src: string; connectionId: string; url: string }>();
  const [failedUrl, setFailedUrl] = useState<string>();
  const preview = src.startsWith('data:image/') || src.startsWith('blob:');

  useEffect(() => {
    if (preview) return;
    const controller = new AbortController();
    let objectUrl: string | undefined;
    async function load() {
      try {
        validateVoiceboxConnection();
        if (new URL(src).origin !== new URL(useServerStore.getState().serverUrl).origin) {
          throw new Error('Image does not belong to the connected server');
        }
        const response = await authenticatedFetch(src, { signal: controller.signal });
        if (!response.ok) throw new Error('Image could not be loaded');
        const bounded = await bufferResponseBounded(response, 5 * 1024 * 1024);
        const blob = await bounded.blob();
        controller.signal.throwIfAborted();
        objectUrl = URL.createObjectURL(blob);
        setLoaded({ src, connectionId, url: objectUrl });
      } catch {
        // Missing avatars and failed connections use the caller's placeholder.
      }
    }
    void load();
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src, connectionId, preview]);

  const url = preview
    ? src
    : loaded?.src === src && loaded.connectionId === connectionId
      ? loaded.url
      : undefined;
  return url && url !== failedUrl ? (
    <img {...props} alt={alt} src={url} onError={() => setFailedUrl(url)} />
  ) : (
    fallback
  );
}
