`image-size@2.0.2.patch` fixes non-advancing ICNS and ISO image box lengths
behind CVE-2025-71330 and CVE-2025-71329. No patched release was available
when audited on 2026-09-05. Zero-sized ISO boxes extend to the end of the
input; undersized boxes and ICNS entries are rejected before iteration.

Run `bun test tests` in `docs/` to verify the malformed image regressions.
Remove the patch once an upstream release includes these fixes and the tests pass.
