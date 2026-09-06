import type { VoiceProfileResponse } from '@/lib/api/types';

export function isFinetunedVoice(profile?: VoiceProfileResponse | null): boolean {
  return (
    profile?.voice_type === 'preset' &&
    profile.preset_engine === 'qwen_custom_voice' &&
    (profile.preset_voice_id?.startsWith('finetuned:') ?? false)
  );
}
