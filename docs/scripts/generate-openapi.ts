import { generateFiles } from 'fumadocs-openapi';
import { openapi } from '../lib/openapi';

await generateFiles({
  input: openapi,
  output: 'content/docs/api-reference',
  // Most runtime routes have no OpenAPI tag. Keep the reference navigable by
  // route family without changing the backend's schema or generation code.
  groupBy(entry) {
    if (entry.type !== 'operation') return 'webhooks';
    const family = entry.item.path.split('/')[1] || 'general';
    const groups: Record<string, string> = {
      health: 'general',
      shutdown: 'general',
      watchdog: 'general',
      generate: 'generation',
      generations: 'generation',
      audio: 'generation',
      transcribe: 'generation',
      samples: 'profiles',
      speak: 'agents',
      mcp: 'agents',
      events: 'agents',
      capture: 'captures',
      cache: 'models',
      backend: 'backends',
    };
    return groups[family] ?? family;
  },
});

console.log('✓ OpenAPI documentation generated in content/docs/api-reference/');
