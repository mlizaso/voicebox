import type { QueryClient } from '@tanstack/react-query';

/** Model changes affect both the library and the recording readiness gates. */
export async function invalidateModelState(
  queryClient: QueryClient,
  refetchType: 'active' | 'all' = 'active',
): Promise<void> {
  await Promise.all([
    queryClient.invalidateQueries({ queryKey: ['modelStatus'], refetchType }),
    queryClient.invalidateQueries({ queryKey: ['capture-readiness'], refetchType }),
  ]);
}
