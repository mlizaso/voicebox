import { PermissionNotice } from '@/components/PermissionNotice';
import { usePlatformPermission } from '@/lib/hooks/usePlatformPermission';

/** Accessibility gates synthetic paste; dictation can still save a capture. */
export function useAccessibilityPermission() {
  return usePlatformPermission('accessibility');
}

export function AccessibilityNotice() {
  const permission = useAccessibilityPermission();
  return <PermissionNotice kind="accessibility" permission={permission} />;
}
