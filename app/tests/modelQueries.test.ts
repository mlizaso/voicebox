import { expect, test } from 'bun:test';
import { QueryClient, QueryObserver } from '@tanstack/react-query';
import { invalidateModelState } from '../src/lib/modelQueries';

test('model changes refresh active readiness even when its ready result is fresh indefinitely', async () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  let installed = true;
  let readinessChecks = 0;
  const observer = new QueryObserver(client, {
    queryKey: ['capture-readiness'],
    queryFn: async () => {
      readinessChecks += 1;
      return { stt: { ready: installed }, llm: { ready: true } };
    },
  });
  const unsubscribe = observer.subscribe(() => {});
  try {
    await observer.refetch();
    expect(observer.getCurrentResult().data?.stt.ready).toBe(true);
    const initialChecks = readinessChecks;
    installed = false;
    await invalidateModelState(client);
    expect(observer.getCurrentResult().data?.stt.ready).toBe(false);
    expect(readinessChecks).toBe(initialChecks + 1);
  } finally {
    unsubscribe();
    client.clear();
  }
});

test('model changes invalidate inactive consumers and honor explicit background refresh', async () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  let installed = true;
  try {
    for (const key of ['modelStatus', 'capture-readiness']) {
      await client.fetchQuery({ queryKey: [key], queryFn: async () => installed });
    }
    installed = false;
    await invalidateModelState(client);
    for (const key of ['modelStatus', 'capture-readiness']) {
      expect(client.getQueryState([key])?.isInvalidated).toBe(true);
      expect(client.getQueryData<boolean>([key])).toBe(true);
    }
    await invalidateModelState(client, 'all');
    for (const key of ['modelStatus', 'capture-readiness']) {
      expect(client.getQueryData<boolean>([key])).toBe(false);
      expect(client.getQueryState([key])?.isInvalidated).toBe(false);
    }
  } finally {
    client.clear();
  }
});
