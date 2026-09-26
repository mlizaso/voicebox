import {
  type MutateOptions,
  type QueryClient,
  useMutation,
  useMutationState,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';
import { useServerStore } from '@/stores/serverStore';

interface SettingsMutation<T> {
  patch: Partial<T>;
  connectionId: string;
}

interface SettingsContext<T> {
  previous: T | undefined;
}

function assertCurrentConnection(connectionId: string): void {
  if (useServerStore.getState().connectionId !== connectionId) {
    throw new Error('Connection changed before settings could be saved');
  }
}

/** Confirmed server state and pending intent have separate owners: failed writes
 * disappear from the overlay without restoring snapshots over newer changes. */
export function useSettingsResource<T extends object>(
  queryKey: readonly string[],
  read: () => Promise<T>,
  write: (patch: Partial<T>) => Promise<T>,
  onSettled?: (client: QueryClient, patch: Partial<T>) => void,
) {
  const queryClient = useQueryClient();
  const connectionId = useServerStore((state) => state.connectionId);
  const mutationKey = [...queryKey, connectionId];
  const query = useQuery({ queryKey, queryFn: read, staleTime: Infinity });
  const mutation = useMutation<T, Error, SettingsMutation<T>, SettingsContext<T>>({
    mutationKey,
    // Settings endpoints apply patches. Preserve submission order on the server
    // even when different mounted controls submit changes at the same time.
    scope: { id: mutationKey.join(':') },
    mutationFn: ({ patch, connectionId: expectedConnection }) => {
      assertCurrentConnection(expectedConnection);
      return write(patch);
    },
    onMutate: async (variables) => {
      // A retained callback must not cancel the newly selected server's read.
      assertCurrentConnection(variables.connectionId);
      await queryClient.cancelQueries({ queryKey });
      return { previous: queryClient.getQueryData<T>(queryKey) };
    },
    onSuccess: (data, variables) => {
      if (variables.connectionId === useServerStore.getState().connectionId) {
        queryClient.setQueryData(queryKey, data);
      }
    },
    onSettled: (_data, error, variables) => {
      if (variables.connectionId !== useServerStore.getState().connectionId) return;
      onSettled?.(queryClient, variables.patch);
      if (
        error &&
        queryClient.getQueryData<T>(queryKey) === undefined &&
        queryClient.isMutating({
          mutationKey: [...queryKey, variables.connectionId],
          exact: true,
          predicate: (pending) =>
            (pending.state.variables as SettingsMutation<T> | undefined)?.connectionId ===
            variables.connectionId,
        }) === 1
      ) {
        // onSettled still counts this mutation as pending. Wait for the last
        // queued write before recovering an initial read that onMutate cancelled.
        void queryClient.invalidateQueries({ queryKey });
      }
    },
  });
  const pendingPatches = useMutationState({
    filters: { mutationKey, exact: true, status: 'pending' },
    select: (pending) => (pending.state.variables as SettingsMutation<T>).patch,
  });
  let settings = query.data;
  if (settings && pendingPatches.length > 0) {
    settings = { ...settings };
    for (const patch of pendingPatches) Object.assign(settings, patch);
  }

  const update = (
    patch: Partial<T>,
    options?: MutateOptions<T, Error, Partial<T>, SettingsContext<T>>,
  ) => {
    mutation.mutate(
      { patch, connectionId },
      options && {
        onSuccess: (data, variables, context, mutationContext) =>
          options.onSuccess?.(data, variables.patch, context, mutationContext),
        onError: (error, variables, context, mutationContext) =>
          options.onError?.(error, variables.patch, context, mutationContext),
        onSettled: (data, error, variables, context, mutationContext) =>
          options.onSettled?.(data, error, variables.patch, context, mutationContext),
      },
    );
  };

  return { settings, isLoading: query.isLoading, update };
}
