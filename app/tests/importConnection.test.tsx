import { afterEach, beforeEach, expect, mock, spyOn, test } from 'bun:test';
import { defaultScheduler, notifyManager, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { I18nextProvider } from 'react-i18next';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import { StoryContent } from '../src/components/StoriesTab/StoryContent';
import * as dialog from '../src/components/ui/dialog';
import { Form } from '../src/components/ui/form';
import * as select from '../src/components/ui/select';
import { ProfileForm } from '../src/components/VoiceProfiles/ProfileForm';
import { apiClient } from '../src/lib/api/client';
import type {
  GenerationResponse,
  StoryItemDetail,
  VoiceProfileResponse,
} from '../src/lib/api/types';
import { queryClient } from '../src/lib/queryClient';
import * as audio from '../src/lib/utils/audio';
import { PlatformProvider } from '../src/platform/PlatformContext';
import type { Platform } from '../src/platform/types';
import { useServerStore } from '../src/stores/serverStore';
import { useStoryStore } from '../src/stores/storyStore';
import { useUIStore } from '../src/stores/uiStore';
import { testI18n } from './testI18n';

let renderer: ReactTestRenderer | undefined;
const originalWindow = globalThis.window;
const originalFrame = globalThis.requestAnimationFrame;
const originalCancelFrame = globalThis.cancelAnimationFrame;
const platform = {
  metadata: { isTauri: false },
  audio: { isSystemAudioSupported: async () => false },
} as unknown as Platform;

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function connect(connectionId: string, serverUrl = 'http://127.0.0.1:17493') {
  return useServerStore.getState().setConnection({
    connectionId,
    serverUrl,
    remoteApiToken: '',
    mode: 'local',
  });
}

beforeEach(() => {
  notifyManager.setScheduler(queueMicrotask);
  queryClient.clear();
  connect('imports-original');
  Object.assign(globalThis, {
    window: { setTimeout, clearTimeout },
    requestAnimationFrame: () => 0,
    cancelAnimationFrame: () => {},
  });
});

afterEach(() => {
  act(() => renderer?.unmount());
  renderer = undefined;
  mock.restore();
  queryClient.clear();
  notifyManager.setScheduler(defaultScheduler);
  useStoryStore.getState().stop();
  useUIStore.setState({ profileDialogOpen: false, profileFormDraft: null, editingProfileId: null });
  Object.assign(globalThis, {
    window: originalWindow,
    requestAnimationFrame: originalFrame,
    cancelAnimationFrame: originalCancelFrame,
  });
});

async function mount(children: ReactNode) {
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={queryClient}>
        <PlatformProvider platform={platform}>
          <I18nextProvider i18n={testI18n}>{children}</I18nextProvider>
        </PlatformProvider>
      </QueryClientProvider>,
    );
  });
}

test.each([
  'replacement',
  'return to original server',
])('a story import stops its batch before adding or uploading after %s', async (end) => {
  spyOn(apiClient, 'getStory').mockResolvedValue({
    id: 'original-story',
    name: 'Story',
    items: [],
    created_at: '',
    updated_at: '',
  });
  spyOn(apiClient, 'listHistory').mockResolvedValue({ items: [], total: 0 });
  const imported = deferred<GenerationResponse>();
  const importAudio = spyOn(apiClient, 'importAudio').mockImplementation(() => imported.promise);
  const addItem = spyOn(apiClient, 'addStoryItem').mockResolvedValue({} as StoryItemDetail);
  useStoryStore.setState({ selectedStoryId: 'original-story' });
  await mount(<StoryContent />);
  act(() => {
    renderer!.root.findByType('input').props.onChange({
      target: { files: [new File(['a'], 'a.wav'), new File(['b'], 'b.wav')], value: '' },
    });
  });
  expect(importAudio).toHaveBeenCalledTimes(1);
  await act(async () => {
    expect(connect('imports-replacement', 'http://127.0.0.1:17494')).toBe(true);
    if (end === 'return to original server') expect(connect('imports-returned')).toBe(true);
    imported.resolve({ id: 'original-audio' } as GenerationResponse);
  });
  expect(addItem).not.toHaveBeenCalled();
  expect(importAudio).toHaveBeenCalledTimes(1);
});

async function mountProfile(sampleFile: File) {
  // Keep the real form/provider and submission path, replacing only its portal.
  spyOn(dialog, 'DialogContent').mockImplementation((({ children }) => (
    <>{children}</>
  )) as typeof dialog.DialogContent);
  spyOn(select, 'SelectContent').mockImplementation(
    ((_props) => null) as typeof select.SelectContent,
  );
  useUIStore.setState({ profileDialogOpen: true, editingProfileId: null, profileFormDraft: null });
  await mount(<ProfileForm />);
  const form = renderer!.root.findByType(Form).props;
  act(() => {
    form.setValue('name', 'Original voice');
    form.setValue('referenceText', 'Original sample');
    form.setValue('sampleFile', sampleFile);
  });
  await act(async () => {});
}

function submitProfile(): Promise<void> {
  return renderer!.root.findByType('form').props.onSubmit({
    preventDefault() {},
    persist() {},
  });
}

test('profile validation cannot create a voice on a connection chosen during decoding', async () => {
  const duration = deferred<number>();
  spyOn(audio, 'getAudioDuration')
    .mockResolvedValueOnce(1)
    .mockImplementation(() => duration.promise);
  const createProfile = spyOn(apiClient, 'createProfile').mockResolvedValue({
    id: 'voice',
  } as VoiceProfileResponse);
  const addSample = spyOn(apiClient, 'addProfileSample').mockResolvedValue({} as never);
  await mountProfile(new File(['sample'], 'sample.wav', { type: 'audio/wav' }));
  let submission!: Promise<void>;
  await act(async () => {
    submission = submitProfile();
  });
  await act(async () => {
    expect(connect('imports-replacement')).toBe(true);
  });
  await act(async () => {
    duration.resolve(1);
    await submission;
  });
  expect(createProfile).not.toHaveBeenCalled();
  expect(addSample).not.toHaveBeenCalled();
});

test.each([
  'resolve',
  'reject',
])('a converted sample cannot upload or roll back on a replacement connection after %s', async (result) => {
  spyOn(audio, 'getAudioDuration').mockResolvedValue(1);
  const converted = deferred<Blob>();
  spyOn(audio, 'convertToWav').mockImplementation(() => converted.promise);
  const createProfile = spyOn(apiClient, 'createProfile').mockResolvedValue({
    id: 'original-voice',
  } as VoiceProfileResponse);
  const addSample = spyOn(apiClient, 'addProfileSample').mockRejectedValue(
    new Error('Sample rejected'),
  );
  const deleteProfile = spyOn(apiClient, 'deleteProfile').mockResolvedValue(undefined);
  await mountProfile(new File(['sample'], 'sample.mp3', { type: 'audio/mpeg' }));
  let submission!: Promise<void>;
  await act(async () => {
    submission = submitProfile();
  });
  expect(createProfile).toHaveBeenCalledTimes(1);
  await act(async () => {
    expect(connect('imports-replacement')).toBe(true);
  });
  await act(async () => {
    if (result === 'resolve') converted.resolve(new Blob(['wav'], { type: 'audio/wav' }));
    else converted.reject(new Error('Unsupported codec'));
    await submission;
  });
  expect(addSample).not.toHaveBeenCalled();
  expect(deleteProfile).not.toHaveBeenCalled();
});

test('profile creation still converts and uploads its sample on an unchanged connection', async () => {
  spyOn(audio, 'getAudioDuration').mockResolvedValue(1);
  spyOn(audio, 'convertToWav').mockResolvedValue(new Blob(['wav'], { type: 'audio/wav' }));
  spyOn(apiClient, 'createProfile').mockResolvedValue({
    id: 'original-voice',
  } as VoiceProfileResponse);
  const addSample = spyOn(apiClient, 'addProfileSample').mockResolvedValue({} as never);
  await mountProfile(new File(['sample'], 'sample.mp3', { type: 'audio/mpeg' }));
  await act(async () => {
    await submitProfile();
  });
  expect(addSample).toHaveBeenCalledWith('original-voice', expect.any(File), 'Original sample');
  const file = addSample.mock.calls[0][1];
  expect(file.name).toBe('sample.wav');
  expect(await file.text()).toBe('wav');
  expect(useUIStore.getState().profileDialogOpen).toBe(false);
});
