#!/usr/bin/env bash
set -euo pipefail

# Optional guardrail: forbid unwrap()/expect() in production code.
#
# Disabled by default to avoid churn; enable by setting:
#   LINT_UNWRAP_EXPECT=1
#
# Scope:
# - crates/cas_cli/src/**/*.rs (excluding test directories/files)

if [[ "${LINT_UNWRAP_EXPECT:-0}" != "1" ]]; then
  echo "ℹ️  unwrap/expect guardrail disabled (set LINT_UNWRAP_EXPECT=1 to enable)."
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/crates/cas_cli/src"

# Use rg if available, otherwise fall back to grep
if command -v rg >/dev/null 2>&1; then
  # Find unwrap()/expect() in non-test code.
  MATCHES=$(rg -n "\.unwrap\(\)|\.expect\(" "$SRC_DIR" \
    --glob '*.rs' \
    --glob '!**/tests/**' \
    --glob '!**/test/**' \
    --glob '!**/*test*.rs' \
    --glob '!**/*tests*.rs' \
    || true)
else
  # grep fallback: find in .rs files excluding test paths
  MATCHES=$(find "$SRC_DIR" -name '*.rs' \
    ! -path '*/tests/*' \
    ! -path '*/test/*' \
    ! -name '*test*.rs' \
    ! -name '*tests*.rs' \
    -exec grep -Hn '\.unwrap()\|\.expect(' {} \; 2>/dev/null || true)
fi

if [[ -n "$MATCHES" ]]; then
  echo "✘ unwrap()/expect() found in cas_cli production code (enable allowlists or refactor)." >&2
  echo "" >&2
  echo "Matches:" >&2
  echo "$MATCHES" >&2
  exit 1
fi

echo "✔ unwrap/expect guardrail: OK"
