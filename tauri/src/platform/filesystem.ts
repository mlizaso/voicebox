import type { FileFilter, PlatformFilesystem } from '@/platform/types';

export const tauriFilesystem: PlatformFilesystem = {
  async saveFile(filename: string, blob: Blob, filters?: FileFilter[]) {
    const { save } = await import('@tauri-apps/plugin-dialog');
    const { writeFile } = await import('@tauri-apps/plugin-fs');

    const filePath = await save({
      defaultPath: filename,
      filters: filters || [],
    });

    if (!filePath) return; // User cancelled the dialog

    const resolvedPath =
      typeof filePath === 'string' ? filePath : (filePath as { path: string }).path;

    if (!resolvedPath) {
      throw new Error('Failed to resolve save path from dialog');
    }

    const arrayBuffer = await blob.arrayBuffer();
    await writeFile(resolvedPath, new Uint8Array(arrayBuffer));
  },

  async saveResponse(
    filename: string,
    getResponse: () => Promise<Response>,
    maxBytes: number,
    filters?: FileFilter[],
  ) {
    if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
      throw new Error('Invalid download size limit');
    }

    const { save } = await import('@tauri-apps/plugin-dialog');
    const filePath = await save({
      defaultPath: filename,
      filters: filters || [],
    });

    if (!filePath) return;

    const resolvedPath =
      typeof filePath === 'string' ? filePath : (filePath as { path: string }).path;
    if (!resolvedPath) {
      throw new Error('Failed to resolve save path from dialog');
    }

    const response = await getResponse();
    const rawLength = response.headers.get('content-length');
    const declaredLength = rawLength === null ? null : Number(rawLength);
    if (
      declaredLength !== null &&
      Number.isSafeInteger(declaredLength) &&
      declaredLength > maxBytes
    ) {
      await response.body?.cancel();
      throw new Error('The exported file exceeds the allowed size');
    }
    if (!response.body) {
      throw new Error('The export response did not contain a downloadable body');
    }

    // The dialog grants filesystem scope to the selected file itself. Stream
    // into that handle; a sibling staging path would fall outside that scope.
    const { open } = await import('@tauri-apps/plugin-fs');
    const output = await open(resolvedPath, { write: true, create: true, truncate: true }).catch(
      async (error: unknown) => {
        await response.body?.cancel();
        throw error;
      },
    );
    const reader = response.body.getReader();
    let receivedBytes = 0;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        receivedBytes += value.byteLength;
        if (receivedBytes > maxBytes) {
          throw new Error('The exported file exceeds the allowed size');
        }

        const written = await output.write(value);
        if (written !== value.byteLength) {
          throw new Error('Could not write the complete exported audio chunk');
        }
      }
    } catch (error) {
      await reader.cancel(error).catch(() => undefined);
      await output.truncate(0).catch(() => undefined);
      throw error;
    } finally {
      reader.releaseLock();
      await output.close();
    }
  },

  async openPath(path: string) {
    const { open } = await import('@tauri-apps/plugin-shell');
    await open(path);
  },

  async pickDirectory(title: string) {
    const { open } = await import('@tauri-apps/plugin-dialog');
    const selected = await open({ directory: true, title });
    if (!selected) return null;
    const dir = typeof selected === 'string' ? selected : (selected as { path: string }).path;
    return dir || null;
  },
};
