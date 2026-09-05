// Fetch latest release information from GitHub
export interface DownloadLinks {
  macArm: string;
  macIntel: string;
  windows: string;
  linux: string;
}

export interface ReleaseInfo {
  version: string;
  downloadLinks: DownloadLinks;
  totalDownloads: number;
}

const GITHUB_REPO = 'jamiepine/voicebox';
const GITHUB_API_BASE = 'https://api.github.com';

// Cache for release info (in-memory cache, resets on server restart)
let cachedReleaseInfo: ReleaseInfo | null = null;
let cacheTimestamp: number = 0;
const CACHE_DURATION = 1000 * 60 * 5; // 5 minutes

// Cache for star count
let cachedStarCount: number | null = null;
let starCacheTimestamp: number = 0;
let releaseRequest: Promise<ReleaseInfo> | null = null;
let starRequest: Promise<number> | null = null;

/**
 * Fetches the latest release from GitHub and extracts download links
 */
export async function getLatestRelease(): Promise<ReleaseInfo> {
  // Return cached data if still valid
  const now = Date.now();
  if (cachedReleaseInfo && now - cacheTimestamp < CACHE_DURATION) {
    return cachedReleaseInfo;
  }
  if (!releaseRequest) {
    releaseRequest = fetchLatestRelease().finally(() => {
      releaseRequest = null;
    });
  }
  return releaseRequest;
}

async function fetchLatestRelease(): Promise<ReleaseInfo> {
  const signal = AbortSignal.timeout(10_000);
  try {
    const response = await fetch(`${GITHUB_API_BASE}/repos/${GITHUB_REPO}/releases/latest`, {
      signal,
      cache: 'no-store',
      headers: {
        Accept: 'application/vnd.github.v3+json',
      },
    });

    if (!response.ok) {
      throw new Error(`GitHub API error: ${response.status}`);
    }

    const release = await response.json();
    const version = release.tag_name;
    const assets = release.assets || [];

    // Extract download links based on file patterns
    const downloadLinks: Partial<DownloadLinks> = {};

    for (const asset of assets) {
      const name = asset.name.toLowerCase();
      const url = asset.browser_download_url;

      // Skip signature files and other non-downloadable files
      if (name.endsWith('.sig') || name.endsWith('.json') || name.endsWith('.txt')) {
        continue;
      }

      if ((name.includes('aarch64') || name.includes('arm64')) && name.endsWith('.dmg')) {
        downloadLinks.macArm = url;
      } else if (name.includes('x64') && name.endsWith('.dmg')) {
        downloadLinks.macIntel = url;
      } else if (name.endsWith('.msi')) {
        downloadLinks.windows = url;
      } else if (name.endsWith('.appimage') || name.endsWith('.deb')) {
        downloadLinks.linux = url;
      }
    }

    // Fetch total downloads across ALL releases
    const totalDownloads = await getTotalDownloads(signal);

    // Fallback: construct URLs if not found in assets
    const baseUrl = `https://github.com/${GITHUB_REPO}/releases/download/${version}`;

    const releaseInfo: ReleaseInfo = {
      version,
      totalDownloads,
      downloadLinks: {
        macArm:
          downloadLinks.macArm || `${baseUrl}/Voicebox_${version.replace('v', '')}_aarch64.dmg`,
        macIntel:
          downloadLinks.macIntel || `${baseUrl}/Voicebox_${version.replace('v', '')}_x64.dmg`,
        windows:
          downloadLinks.windows || `${baseUrl}/voicebox_${version.replace('v', '')}_x64_en-US.msi`,
        linux: downloadLinks.linux || `${baseUrl}/voicebox_x86_64-unknown-linux-gnu.AppImage`,
      },
    };

    // Update cache
    cachedReleaseInfo = releaseInfo;
    cacheTimestamp = Date.now();

    return releaseInfo;
  } catch (error) {
    console.error('Failed to fetch latest release:', error);
    if (cachedReleaseInfo) return cachedReleaseInfo;
    throw error;
  }
}

/**
 * Fetches download counts across ALL releases (paginated)
 */
async function getTotalDownloads(signal: AbortSignal): Promise<number> {
  let total = 0;
  let page = 1;

  while (true) {
    const response = await fetch(
      `${GITHUB_API_BASE}/repos/${GITHUB_REPO}/releases?per_page=100&page=${page}`,
      {
        signal,
        cache: 'no-store',
        headers: { Accept: 'application/vnd.github.v3+json' },
      },
    );

    if (!response.ok) throw new Error(`GitHub API error: ${response.status}`);

    const releases = await response.json();
    if (!Array.isArray(releases)) throw new Error('Invalid GitHub release list');

    for (const release of releases) {
      for (const asset of release.assets || []) {
        total += asset.download_count || 0;
      }
    }

    if (releases.length < 100) return total;
    page++;
  }
}

/**
 * Fetches the star count for the repo from GitHub
 */
export async function getStarCount(): Promise<number> {
  const now = Date.now();
  if (cachedStarCount !== null && now - starCacheTimestamp < CACHE_DURATION) {
    return cachedStarCount;
  }
  if (!starRequest) {
    starRequest = fetchStarCount().finally(() => {
      starRequest = null;
    });
  }
  return starRequest;
}

async function fetchStarCount(): Promise<number> {
  try {
    const response = await fetch(`${GITHUB_API_BASE}/repos/${GITHUB_REPO}`, {
      signal: AbortSignal.timeout(10_000),
      next: { revalidate: 600 },
      headers: {
        Accept: 'application/vnd.github.v3+json',
      },
    });

    if (!response.ok) {
      throw new Error(`GitHub API error: ${response.status}`);
    }

    const repo = await response.json();
    const count = repo.stargazers_count ?? 0;

    cachedStarCount = count;
    starCacheTimestamp = Date.now();

    return count;
  } catch (error) {
    console.error('Failed to fetch star count:', error);
    if (cachedStarCount !== null) return cachedStarCount;
    throw error;
  }
}
