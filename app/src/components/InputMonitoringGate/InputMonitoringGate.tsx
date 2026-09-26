import { PermissionNotice } from '@/components/PermissionNotice';
import { usePlatformPermission } from '@/lib/hooks/usePlatformPermission';

/** Input Monitoring must be granted before global shortcuts can record. */
export function useInputMonitoringPermission() {
  return usePlatformPermission('inputMonitoring');
}

export function InputMonitoringNotice({ enabled }: { enabled: boolean }) {
  const permission = useInputMonitoringPermission();
  return <PermissionNotice kind="inputMonitoring" permission={permission} enabled={enabled} />;
}
