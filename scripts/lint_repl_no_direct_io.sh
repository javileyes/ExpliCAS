#!/usr/bin/env bash
set -euo pipefail

# Guardrail: forbid direct stdout/stderr I/O in the REPL implementation.
#
# Policy:
# - `println!` / `eprintln!` are allowed ONLY in:
#     * crates/cas_cli/src/repl/init.rs        (inside fn print_reply)
#     * crates/cas_cli/src/repl/show_steps.rs (deferred migration)
# - Everything else under crates/cas_cli/src/repl must avoid direct I/O.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPL_DIR="$ROOT_DIR/crates/cas_cli/src/repl"
INIT_RS="$REPL_DIR/init.rs"
SHOW_STEPS_RS="$REPL_DIR/show_steps.rs"

# Use rg if available, otherwise fall back to grep
if command -v rg >/dev/null 2>&1; then
  USE_RG=1
else
  USE_RG=0
fi

# 1) Hard fail: any println/eprintln in repl/ except init.rs and show_steps.rs.
if [[ $USE_RG -eq 1 ]]; then
  DISALLOWED=$(rg -n "(e)?println!\s*\(" "$REPL_DIR" \
    --glob '!show_steps.rs' \
    --glob '!init.rs' \
    || true)
else
  # grep fallback: search in all .rs files except init.rs and show_steps.rs
  DISALLOWED=$(find "$REPL_DIR" -name '*.rs' ! -name 'init.rs' ! -name 'show_steps.rs' -exec grep -Hn 'println!\|eprintln!' {} \; 2>/dev/null || true)
fi

if [[ -n "$DISALLOWED" ]]; then
  echo "✘ Direct I/O macros are forbidden in repl/ (except init.rs::print_reply and show_steps.rs)." >&2
  echo "" >&2
  echo "Matches:" >&2
  echo "$DISALLOWED" >&2
  exit 1
fi

# 2) In init.rs, only allow println/eprintln inside fn print_reply.
if [[ ! -f "$INIT_RS" ]]; then
  echo "✘ Expected file not found: $INIT_RS" >&2
  exit 1
fi

if [[ $USE_RG -eq 1 ]]; then
  START_LINE=$(rg -n "\bfn\s+print_reply\b" "$INIT_RS" | head -n1 | cut -d: -f1 || true)
else
  START_LINE=$(grep -n 'fn print_reply' "$INIT_RS" | head -n1 | cut -d: -f1 || true)
fi

if [[ -z "$START_LINE" ]]; then
  echo "✘ Could not find fn print_reply in $INIT_RS" >&2
  exit 1
fi

# Find the end of the print_reply function by counting braces
# This awk script handles macOS awk compatibility
END_LINE=$(awk -v start="$START_LINE" '
  NR < start { next }
  NR == start { in_fn=1; depth=0; seen_open=0 }
  in_fn {
    for (i=1; i<=length($0); i++) {
      c = substr($0, i, 1)
      if (c == "{") { depth++; seen_open=1 }
      if (c == "}") { depth-- }
    }
    if (seen_open && depth == 0 && NR > start) { print NR; exit }
  }
  END { if (in_fn) print NR }
' "$INIT_RS" | head -n1)

if [[ $USE_RG -eq 1 ]]; then
  IO_LINES=$(rg -n "(e)?println!\s*\(" "$INIT_RS" | cut -d: -f1 || true)
else
  IO_LINES=$(grep -n 'println!\|eprintln!' "$INIT_RS" | cut -d: -f1 || true)
fi

if [[ -n "$IO_LINES" ]]; then
  while IFS= read -r ln; do
    # Skip empty lines
    [[ -z "$ln" ]] && continue
    if [[ "$ln" -lt "$START_LINE" || "$ln" -gt "$END_LINE" ]]; then
      echo "✘ init.rs contains println/eprintln outside fn print_reply (line $ln)." >&2
      echo "  Allowed range: $START_LINE..$END_LINE" >&2
      grep -n 'println!\|eprintln!' "$INIT_RS" >&2 || true
      exit 1
    fi
  done <<< "$IO_LINES"
fi

echo "✔ repl I/O guardrail: OK (only init.rs::print_reply + show_steps.rs may print)"
