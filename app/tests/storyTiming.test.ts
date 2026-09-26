import { afterEach, expect, test } from 'bun:test';
import type { StoryItemDetail } from '../src/lib/api/types';
import {
  getEffectiveDurationMs,
  getItemEndMs,
  getStoryDurationMs,
  isStoryItemActive,
} from '../src/lib/story/timing';
import { useStoryStore } from '../src/stores/storyStore';

afterEach(() => useStoryStore.getState().stop());

test.each([
  { duration: 2, trim_start_ms: 0, trim_end_ms: 0, effective: 2000 },
  { duration: 2, trim_start_ms: 250, trim_end_ms: 500, effective: 1250 },
  { duration: 0.125, trim_start_ms: 5, trim_end_ms: 20, effective: 100 },
  { duration: 1, trim_start_ms: 500, trim_end_ms: 500, effective: 0 },
])('trimmed $duration-second audio occupies $effective milliseconds', (item) => {
  expect(getEffectiveDurationMs(item)).toBe(item.effective);
  expect(getItemEndMs({ ...item, start_time_ms: 1500 })).toBe(1500 + item.effective);
});

test('adjacent clips use an exclusive trimmed end for highlighting and playback', () => {
  const clip = { duration: 2, trim_start_ms: 250, trim_end_ms: 500, start_time_ms: 1000 };
  expect(isStoryItemActive(clip, 999)).toBe(false);
  expect(isStoryItemActive(clip, 1000)).toBe(true);
  expect(isStoryItemActive(clip, 2249)).toBe(true);
  expect(isStoryItemActive(clip, 2250)).toBe(false);
});

test('story duration includes gaps and overlap and bounds playback seeks', () => {
  const clips = [
    { duration: 2, trim_start_ms: 250, trim_end_ms: 500, start_time_ms: 1000 },
    { duration: 1, trim_start_ms: 0, trim_end_ms: 100, start_time_ms: 8000 },
    { duration: 0.5, trim_start_ms: 0, trim_end_ms: 0, start_time_ms: 8100 },
  ] as StoryItemDetail[];
  expect(getStoryDurationMs([])).toBe(0);
  expect(getStoryDurationMs(clips)).toBe(8900);
  useStoryStore.getState().play('story', clips);
  expect(useStoryStore.getState().totalDurationMs).toBe(8900);
  useStoryStore.getState().seek(10000);
  expect(useStoryStore.getState().currentTimeMs).toBe(8900);
  useStoryStore.getState().pause();
  useStoryStore.getState().seek(8500);
  useStoryStore.getState().play('story', clips);
  expect(useStoryStore.getState().currentTimeMs).toBe(8500);
});
