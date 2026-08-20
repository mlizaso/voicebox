import type { FileFilter, PlatformFilesystem } from '@/platform/types';

const MAX_BUFFERED_DOWNLOAD_BYTES = 64 * 1024 * 1024;

interface BrowserWritableFile {
  write(data: Uint8Array): Promise<void>;
  close(): Promise<void>;
  abort(reason?: unknown): Promise<void>;
}

interface BrowserFileHandle {
  createWritable(): Promise<BrowserWritableFile>;
}

interface SavePickerWindow extends Window {
  showSaveFilePicker?: (options: { suggestedName: string }) => Promise<BrowserFileHandle>;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

function declaredResponseLength(response: Response): number | null {
  const rawLength = response.headers.get('content-length');
  if (rawLength === null || rawLength.trim() === '') return null;
  const length = Number(rawLength);
  return Number.isSafeInteger(length) && length >= 0 ? length : null;
}

export const webFilesystem: PlatformFilesystem = {
  async saveFile(filename: string, blob: Blob, _filters?: FileFilter[]) {
    // Browser: trigger download
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  },

  async saveResponse(
    filename: string,
    getResponse: () => Promise<Response>,
    maxBytes: number,
    _filters?: FileFilter[],
  ) {
    if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
      throw new Error('Invalid download size limit');
    }

    const pickerWindow = window as SavePickerWindow;
    let handle: BrowserFileHandle | null = null;
    if (pickerWindow.showSaveFilePicker) {
      try {
        handle = await pickerWindow.showSaveFilePicker({ suggestedName: filename });
      } catch (error) {
        if (isAbortError(error)) return;
        throw error;
      }
    }

    // Select the destination before starting a potentially long render. This
    // keeps the web picker inside the click's user-activation window and makes
    // cancellation free on both platforms.
    const response = await getResponse();
    const declaredLength = declaredResponseLength(response);
    if (declaredLength !== null && declaredLength > maxBytes) {
      await response.body?.cancel();
      throw new Error('The exported file exceeds the allowed size');
    }
    if (!response.body) {
      throw new Error('The export response did not contain a downloadable body');
    }

    if (handle) {
      let output: BrowserWritableFile;
      try {
        output = await handle.createWritable();
      } catch (error) {
        await response.body.cancel();
        throw error;
      }
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
          await output.write(value);
        }
        await output.close();
      } catch (error) {
        await reader.cancel(error).catch(() => undefined);
        await output.abort(error).catch(() => undefined);
        throw error;
      } finally {
        reader.releaseLock();
      }
      return;
    }

    // Browsers without the File System Access API cannot stream a fetch body
    // into a user-selected file. Keep their compatibility fallback bounded.
    if (declaredLength !== null && declaredLength > MAX_BUFFERED_DOWNLOAD_BYTES) {
      await response.body.cancel();
      throw new Error(
        'This browser cannot save large exports safely. Use the desktop app or a Chromium-based browser.',
      );
    }

    const reader = response.body.getReader();
    const chunks: ArrayBuffer[] = [];
    let receivedBytes = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        receivedBytes += value.byteLength;
        if (receivedBytes > Math.min(maxBytes, MAX_BUFFERED_DOWNLOAD_BYTES)) {
          throw new Error(
            'This browser cannot save large exports safely. Use the desktop app or a Chromium-based browser.',
          );
        }
        const chunk = new ArrayBuffer(value.byteLength);
        new Uint8Array(chunk).set(value);
        chunks.push(chunk);
      }
    } catch (error) {
      await reader.cancel(error).catch(() => undefined);
      throw error;
    } finally {
      reader.releaseLock();
    }

    const blob = new Blob(chunks, { type: response.headers.get('content-type') || 'audio/wav' });
    await this.saveFile(filename, blob);
  },

  async openPath(_path: string) {
    // No filesystem access in browser
  },

  async pickDirectory(_title: string) {
    return null;
  },
};
