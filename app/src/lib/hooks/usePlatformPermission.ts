import { useCallback, useEffect, useRef, useState } from 'react';
import { usePlatform } from '@/platform/PlatformContext';

export type PlatformPermission = 'accessibility' | 'inputMonitoring';

/** Own permission checks, including focus refresh and late IPC responses. */
export function usePlatformPermission(kind: PlatformPermission) {
  const platform = usePlatform();
  const [needsPermission, setNeedsPermission] = useState(platform.metadata.isTauri);
  const [checking, setChecking] = useState(false);
  const active = useRef(true);
  const request = useRef(0);

  const recheck = useCallback(async (): Promise<boolean> => {
    const current = ++request.current;
    setChecking(true);
    try {
      const trusted = await (kind === 'accessibility'
        ? platform.dictation.checkAccessibilityPermission()
        : platform.dictation.checkInputMonitoringPermission());
      if (!active.current || current !== request.current) return false;
      setNeedsPermission(!trusted);
      return trusted;
    } catch (error) {
      if (active.current && current === request.current) {
        setNeedsPermission(true);
        console.warn(`[${kind}] check failed:`, error);
      }
      return false;
    } finally {
      if (active.current && current === request.current) setChecking(false);
    }
  }, [kind, platform.dictation]);

  useEffect(() => {
    active.current = true;
    setNeedsPermission(platform.metadata.isTauri);
    void recheck();
    const onFocus = () => {
      void recheck();
    };
    window.addEventListener('focus', onFocus);
    const unsubscribe =
      kind === 'accessibility'
        ? platform.events.subscribe('system:accessibility-missing', () => {
            // A failed paste is newer evidence than any pending permission query.
            request.current += 1;
            setNeedsPermission(true);
            setChecking(false);
          })
        : undefined;
    return () => {
      active.current = false;
      request.current += 1;
      window.removeEventListener('focus', onFocus);
      unsubscribe?.();
    };
  }, [kind, platform.events, platform.metadata.isTauri, recheck]);

  const openSettings = useCallback(async () => {
    try {
      if (kind === 'accessibility') await platform.dictation.openAccessibilitySettings();
      else await platform.dictation.openInputMonitoringSettings();
    } catch (error) {
      console.warn(`[${kind}] open settings failed:`, error);
    }
  }, [kind, platform.dictation]);

  return { needsPermission, checking, recheck, openSettings };
}
