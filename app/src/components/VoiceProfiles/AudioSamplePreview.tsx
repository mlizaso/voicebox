import { Mic, Pause, Play } from 'lucide-react';
import type { ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';

interface AudioSamplePreviewProps {
  file: File;
  icon: ReactNode;
  label: string;
  actionLabel: string;
  onAction: () => void;
  onPlayPause: () => void;
  onTranscribe: () => void;
  isPlaying: boolean;
  isTranscribing: boolean;
  playbackDisabled?: boolean;
  transcriptionDisabled?: boolean;
}

/** Common completed-sample controls for uploads, microphone and system captures. */
export function AudioSamplePreview({
  file,
  icon,
  label,
  actionLabel,
  onAction,
  onPlayPause,
  onTranscribe,
  isPlaying,
  isTranscribing,
  playbackDisabled = false,
  transcriptionDisabled = false,
}: AudioSamplePreviewProps) {
  const { t } = useTranslation();
  return (
    <>
      <div className="flex items-center gap-2">
        {icon}
        <span className="font-medium">{label}</span>
      </div>
      <p className="text-sm text-muted-foreground text-center">
        {t('audioSample.fileLabel', { name: file.name })}
      </p>
      <div className="flex gap-2">
        <Button
          type="button"
          size="icon"
          variant="outline"
          onClick={onPlayPause}
          disabled={playbackDisabled}
          aria-label={isPlaying ? t('audioSample.pause') : t('audioSample.play')}
        >
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={onTranscribe}
          disabled={isTranscribing || transcriptionDisabled}
          className="flex items-center gap-2"
        >
          <Mic className="h-4 w-4" />
          {isTranscribing ? t('audioSample.transcribing') : t('audioSample.transcribe')}
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={onAction}
          className="flex items-center gap-2"
        >
          {actionLabel}
        </Button>
      </div>
    </>
  );
}
