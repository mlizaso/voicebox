import { AlertTriangle, ExternalLink } from 'lucide-react';
import { useCallback, useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import type { PlatformPermission, usePlatformPermission } from '@/lib/hooks/usePlatformPermission';

/** Common permission guidance; callers retain their own visibility policy. */
export function PermissionNotice({
  kind,
  permission,
  enabled = true,
}: {
  kind: PlatformPermission;
  permission: ReturnType<typeof usePlatformPermission>;
  enabled?: boolean;
}) {
  const { t } = useTranslation();
  const { needsPermission, checking, recheck, openSettings } = permission;
  const [stillMissing, setStillMissing] = useState(false);
  const handleRecheck = useCallback(async () => {
    setStillMissing(false);
    const trusted = await recheck();
    if (!trusted) setStillMissing(true);
  }, [recheck]);

  if (!enabled || !needsPermission) return null;
  const key = `captures.permissions.${kind}`;

  return (
    <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3.5 py-3">
      <div className="flex items-start gap-3">
        <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-amber-500" />
        <div className="flex-1 min-w-0 space-y-1">
          <p className="text-sm font-medium text-foreground">{t(`${key}.title`)}</p>
          <p className="text-sm text-muted-foreground leading-relaxed">
            <Trans i18nKey={`${key}.body`} components={{ path: <span /> }} />
          </p>
          <div className="flex items-center gap-2 pt-1.5">
            <Button type="button" size="sm" onClick={openSettings} className="gap-1.5">
              <ExternalLink className="h-3.5 w-3.5" />
              {t(`${key}.openSettings`)}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleRecheck}
              disabled={checking}
            >
              {checking ? t(`${key}.rechecking`) : t(`${key}.recheck`)}
            </Button>
          </div>
          {stillMissing && !checking && (
            <p className="text-xs text-amber-600 dark:text-amber-400 pt-1">
              {t(`${key}.stillMissing`)}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
