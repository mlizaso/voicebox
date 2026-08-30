import { QueryClientProvider } from '@tanstack/react-query';
import { StrictMode } from 'react';
import App from './App';
import './i18n';
import { queryClient } from './lib/queryClient';
import { PlatformProvider } from './platform/PlatformContext';
import type { Platform } from './platform/types';

export interface AppRootProps {
  platform: Platform;
}

/** Shared provider composition for every Voicebox frontend runtime. */
export function AppRoot({ platform }: AppRootProps) {
  return (
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <PlatformProvider platform={platform}>
          <App />
        </PlatformProvider>
      </QueryClientProvider>
    </StrictMode>
  );
}
