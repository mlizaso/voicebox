import type { Platform } from '@/platform/types';
import { webAudio } from './audio';
import { webDictation } from './dictation';
import { webEvents } from './events';
import { webFilesystem } from './filesystem';
import { webLifecycle } from './lifecycle';
import { webMetadata } from './metadata';
import { webUpdater } from './updater';

export const webPlatform: Platform = {
  filesystem: webFilesystem,
  events: webEvents,
  dictation: webDictation,
  updater: webUpdater,
  audio: webAudio,
  lifecycle: webLifecycle,
  metadata: webMetadata,
};
