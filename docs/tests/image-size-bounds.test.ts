import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';

// Each exploit used to lock the parser's event loop. Isolate it so a regression
// fails with a timeout instead of hanging the documentation build's test run.
test.each(['icns', 'jxl', 'heif'])('%s rejects non-advancing box lengths', (format) => {
  const code = `
    const { imageSize } = require('image-size');
    const box = (name, data, length) => {
      const result = Buffer.alloc(8 + data.length);
      result.writeUInt32BE(length ?? result.length);
      result.write(name, 4);
      data.copy(result, 8);
      return result;
    };
    const fixtures = {
      icns: Buffer.from('69636e73000000106963303700000000', 'hex'),
      jxl: Buffer.concat([Buffer.from('0000000c4a584c200d0a870a', 'hex'), box('ftyp', Buffer.from('jxl ')), box('jxlp', Buffer.alloc(4), 0)]),
      heif: Buffer.concat([box('ftyp', Buffer.from('heic0000')), box('meta', Buffer.concat([Buffer.alloc(4), box('iprp', box('ipco', box('ispe', Buffer.alloc(12), 0))) ]))]),
    };
    try { imageSize(fixtures[${JSON.stringify(format)}]); }
    catch (error) { if (!(error instanceof Error)) process.exit(2); }
  `;
  const result = spawnSync(process.execPath, ['-e', code], {
    cwd: `${import.meta.dir}/..`,
    timeout: 1000,
    encoding: 'utf8',
  });
  expect(result.error).toBeUndefined();
  expect(result.status).toBe(0);
});
