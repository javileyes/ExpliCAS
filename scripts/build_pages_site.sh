#!/usr/bin/env bash
# Build the static GitHub Pages site (frente W · W4): the full ExpliCAS web
# UI running the engine locally in the browser via WebAssembly.
#
# Usage: scripts/build_pages_site.sh [out_dir]   (default: dist/pages)
#
# Requirements:
# - rustup with the NIGHTLY toolchain + wasm32-unknown-unknown target.
#   Nightly is REQUIRED for the wasm build only (W3 finding: stable's LLVM
#   explodes in memory doing wasm32 codegen of cas_engine — fixed upstream);
#   native builds stay on stable, untouched.
# - wasm-pack.
#
# The server deployment is NOT affected by any of this: web/server.py keeps
# serving the same files with engineMode unset (server mode).

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-dist/pages}"

echo "==> wasm build (nightly, opt-level=s for size)"
RUSTUP_TOOLCHAIN=nightly \
CARGO_PROFILE_RELEASE_OPT_LEVEL=s \
CARGO_PROFILE_RELEASE_LTO=false \
    wasm-pack build crates/cas_wasm --target web --release --out-dir pkg

echo "==> assemble site at $OUT"
rm -rf "$OUT"
mkdir -p "$OUT/api"
cp web/index.html web/wasm-mode.js web/wasm_worker.js "$OUT/"
cp -r crates/cas_wasm/pkg "$OUT/pkg"
rm -f "$OUT/pkg/.gitignore"

# Pages/WASM build config: activates the dual-mode shim.
cat > "$OUT/build-config.js" <<'EOF'
// GitHub Pages build: the engine runs fully in the browser (WASM mode).
window.EXPLICAS_BUILD_CONFIG = Object.freeze({"defaultTimeBudgetMs": 2700, "engineMode": "wasm"});
EOF

# Static /api/examples (same JSON shape server.py serves), so the Ejemplos
# button works without a backend.
python3 - "$OUT/api/examples" <<'EOF'
import csv, json, sys
out = sys.argv[1]
examples = []
with open('web/examples.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        examples.append({
            "expression": row.get('expression', ''),
            "description": row.get('description', ''),
            "group": row.get('group', ''),
        })
with open(out, 'w', encoding='utf-8') as f:
    json.dump({"ok": True, "examples": examples}, f, ensure_ascii=False)
print(f"api/examples: {len(examples)} rows")
EOF

# Pages hygiene: disable Jekyll processing.
touch "$OUT/.nojekyll"

echo "==> done"
ls -la "$OUT"
du -h "$OUT/pkg/cas_wasm_bg.wasm"
