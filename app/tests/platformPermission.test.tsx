import { afterEach, expect, mock, test } from 'bun:test';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import { useAccessibilityPermission } from '../src/components/AccessibilityGate/AccessibilityGate';
import { useInputMonitoringPermission } from '../src/components/InputMonitoringGate/InputMonitoringGate';
import { PlatformProvider } from '../src/platform/PlatformContext';
import type { Platform } from '../src/platform/types';

const originalWindow = globalThis.window;
let renderer: ReactTestRenderer | undefined;

afterEach(() => {
  act(() => renderer?.unmount());
  renderer = undefined;
  globalThis.window = originalWindow;
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((accept, fail) => {
    resolve = accept;
    reject = fail;
  });
  return { promise, resolve, reject };
}

for (const usePermission of [useAccessibilityPermission, useInputMonitoringPermission]) {
  test(`${usePermission.name} accepts a grant and rechecks it on window focus`, async () => {
    let granted = true;
    const check = mock(async () => granted);
    const platform = {
      metadata: { isTauri: true },
      dictation: { checkAccessibilityPermission: check, checkInputMonitoringPermission: check },
      events: { subscribe: () => () => {} },
    } as unknown as Platform;
    globalThis.window = new EventTarget() as unknown as Window & typeof globalThis;
    let permission!: ReturnType<typeof usePermission>;
    function Probe() {
      permission = usePermission();
      return null;
    }
    await act(async () => {
      renderer = create(
        <PlatformProvider platform={platform}>
          <Probe />
        </PlatformProvider>,
      );
    });
    expect(permission.needsPermission).toBe(false);
    expect(permission.checking).toBe(false);
    granted = false;
    await act(async () => {
      window.dispatchEvent(new Event('focus'));
    });
    expect(permission.needsPermission).toBe(true);
    expect(check).toHaveBeenCalledTimes(2);
  });

  test(`${usePermission.name} requires a successful check before granting permission`, async () => {
    const pending = deferred<boolean>();
    const check = mock(() => pending.promise);
    const platform = {
      metadata: { isTauri: true },
      dictation: { checkAccessibilityPermission: check, checkInputMonitoringPermission: check },
      events: { subscribe: () => () => {} },
    } as unknown as Platform;
    globalThis.window = new EventTarget() as unknown as Window & typeof globalThis;
    let permission!: ReturnType<typeof usePermission>;
    function Probe() {
      permission = usePermission();
      return null;
    }
    await act(async () => {
      renderer = create(
        <PlatformProvider platform={platform}>
          <Probe />
        </PlatformProvider>,
      );
    });
    expect(permission.needsPermission).toBe(true);
    expect(permission.checking).toBe(true);
    await act(async () => pending.reject(new Error('IPC unavailable')));
    expect(permission.needsPermission).toBe(true);
    expect(permission.checking).toBe(false);
  });

  test(`${usePermission.name} ignores a stale successful check after a newer denial`, async () => {
    const first = deferred<boolean>();
    const second = deferred<boolean>();
    const check = mock(() => first.promise)
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);
    const platform = {
      metadata: { isTauri: true },
      dictation: { checkAccessibilityPermission: check, checkInputMonitoringPermission: check },
      events: { subscribe: () => () => {} },
    } as unknown as Platform;
    globalThis.window = new EventTarget() as unknown as Window & typeof globalThis;
    let permission!: ReturnType<typeof usePermission>;
    function Probe() {
      permission = usePermission();
      return null;
    }
    await act(async () => {
      renderer = create(
        <PlatformProvider platform={platform}>
          <Probe />
        </PlatformProvider>,
      );
    });
    let recheck!: Promise<boolean>;
    act(() => {
      recheck = permission.recheck();
    });
    await act(async () => {
      second.resolve(false);
      await recheck;
    });
    await act(async () => first.resolve(true));
    expect(permission.needsPermission).toBe(true);
  });
}

test('an accessibility-missing event supersedes an in-flight successful check', async () => {
  const pending = deferred<boolean>();
  let reportMissing!: () => void;
  const platform = {
    metadata: { isTauri: true },
    dictation: { checkAccessibilityPermission: () => pending.promise },
    events: {
      subscribe: (_name: string, callback: () => void) => {
        reportMissing = callback;
        return () => {};
      },
    },
  } as unknown as Platform;
  globalThis.window = new EventTarget() as unknown as Window & typeof globalThis;
  let permission!: ReturnType<typeof useAccessibilityPermission>;
  function Probe() {
    permission = useAccessibilityPermission();
    return null;
  }
  await act(async () => {
    renderer = create(
      <PlatformProvider platform={platform}>
        <Probe />
      </PlatformProvider>,
    );
  });
  act(() => reportMissing());
  await act(async () => pending.resolve(true));
  expect(permission.needsPermission).toBe(true);
  expect(permission.checking).toBe(false);
});
