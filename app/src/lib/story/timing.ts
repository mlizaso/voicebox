import type { StoryItemDetail } from '@/lib/api/types';

type TrimmedAudio = Pick<StoryItemDetail, 'duration' | 'trim_start_ms' | 'trim_end_ms'>;
type TimedAudio = TrimmedAudio & Pick<StoryItemDetail, 'start_time_ms'>;

/** API durations are seconds; trims and all story timeline coordinates are milliseconds. */
export function getEffectiveDurationMs(item: TrimmedAudio): number {
  return item.duration * 1000 - (item.trim_start_ms || 0) - (item.trim_end_ms || 0);
}

export function getItemEndMs(item: TimedAudio): number {
  return item.start_time_ms + getEffectiveDurationMs(item);
}

/** The end boundary is exclusive so adjacent clips never overlap for one frame. */
export function isStoryItemActive(item: TimedAudio, timeMs: number): boolean {
  return timeMs >= item.start_time_ms && timeMs < getItemEndMs(item);
}

export function getStoryDurationMs(items: readonly TimedAudio[]): number {
  return items.reduce((end, item) => Math.max(end, getItemEndMs(item)), 0);
}
