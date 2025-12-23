#!/usr/bin/env bash
set -euo pipefail

# scripts/ci.sh — Local CI runner
#
# What it can run:
#   - cargo fmt --check
#   - repo lints (scripts/lint_*.sh)
#   - cargo clippy -D warnings
#   - cargo test (debug)
#   - cargo build --release
#   - cargo test --release (optional; slower)
#
# Usage:
#   ./scripts/ci.sh                   # full (fmt+lints+clippy+tests) on pinned toolchain
#   ./scripts/ci.sh --release-build   # additionally build --release
#   ./scripts/ci.sh --release-test    # additionally test --release
#   ./scripts/ci.sh --msrv            # run the same suite also on MSRV (if configured)
#   ./scripts/ci.sh --quick           # skip clippy
#   ./scripts/ci.sh --lint            # fmt + lints + clippy (no tests)
#   ./scripts/ci.sh --test            # tests only
#   ./scripts/ci.sh --toolchain 1.81.0
#
# Notes:
# - Toolchain pinned via rust-toolchain.toml (channel = "x.y.z") if present
# - MSRV read from root Cargo.toml rust-version if present
# - If lints are enabled and scripts use rg, you need ripgrep installed.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -----------------------------
# pretty output helpers
# -----------------------------
RED=$'\033[31m'
GRN=$'\033[32m'
YLW=$'\033[33m'
BLU=$'\033[34m'
RST=$'\033[0m'

step() { echo "${BLU}==>${RST} $*"; }
ok()   { echo "${GRN}✔${RST} $*"; }
warn() { echo "${YLW}⚠${RST} $*"; }
die()  { echo "${RED}✘${RST} $*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

read_toolchain_from_rust_toolchain_toml() {
  local f="$ROOT_DIR/rust-toolchain.toml"
  [[ -f "$f" ]] || { echo ""; return; }
  sed -nE 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"([^"]+)".*$/\1/p' "$f" | head -n1
}

read_msrv_from_cargo_toml() {
  local f="$ROOT_DIR/Cargo.toml"
  [[ -f "$f" ]] || { echo ""; return; }
  sed -nE 's/^[[:space:]]*rust-version[[:space:]]*=[[:space:]]*"([^"]+)".*$/\1/p' "$f" | head -n1
}

ensure_toolchain_installed() {
  local tc="$1"
  [[ -n "$tc" ]] || return 0
  if ! rustup toolchain list | awk '{print $1}' | grep -qx "$tc"; then
    step "Installing Rust toolchain $tc (via rustup)"
    rustup toolchain install "$tc" >/dev/null
  fi
}

run() {
  local name="$1"; shift
  step "$name"
  (cd "$ROOT_DIR" && "$@")
  ok "$name"
}

# -----------------------------
# defaults / flags
# -----------------------------
RUN_FMT=1
RUN_CLIPPY=1
RUN_TESTS=1
RUN_LINTS=1
RUN_RELEASE_BUILD=0
RUN_RELEASE_TEST=0

QUICK=0
RUN_MSRV=0
OVERRIDE_TOOLCHAIN=""

# test args (debug) + clippy args
CARGO_TEST_ARGS=(test --workspace)
CLIPPY_ARGS=(clippy --workspace --all-targets -- -D warnings)

show_help() {
  sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --msrv) RUN_MSRV=1 ;;
    --toolchain) shift; OVERRIDE_TOOLCHAIN="${1:-}" ;;
    --no-fmt) RUN_FMT=0 ;;
    --no-clippy) RUN_CLIPPY=0 ;;
    --no-tests) RUN_TESTS=0 ;;
    --no-lints) RUN_LINTS=0 ;;

    --lint) RUN_TESTS=0 ;;
    --test) RUN_FMT=0; RUN_CLIPPY=0; RUN_LINTS=0 ;;

    --release-build) RUN_RELEASE_BUILD=1 ;;
    --release-test) RUN_RELEASE_TEST=1 ;;
    # Backward-compat alias:
    --release) RUN_RELEASE_TEST=1 ;;

    --all-features) CARGO_TEST_ARGS+=(--all-features) ;;
    --features) shift; CARGO_TEST_ARGS+=(--features "${1:-}") ;;

    -h|--help) show_help; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
  shift
done

if [[ "$QUICK" -eq 1 ]]; then
  RUN_CLIPPY=0
fi

need_cmd rustup
need_cmd cargo

PINNED_TC="$(read_toolchain_from_rust_toolchain_toml)"
MSRV="$(read_msrv_from_cargo_toml)"

TOOLCHAIN="${OVERRIDE_TOOLCHAIN:-$PINNED_TC}"
if [[ -z "$TOOLCHAIN" ]]; then
  warn "No rust-toolchain.toml found (or no channel). Using default 'stable'."
  TOOLCHAIN="stable"
fi

if [[ -z "$MSRV" ]]; then
  warn "No rust-version found in root Cargo.toml. MSRV run will be skipped unless set."
fi

ensure_toolchain_installed "$TOOLCHAIN"

echo "${BLU}Local CI Runner${RST}"
echo "  Repo:           $ROOT_DIR"
echo "  Toolchain:      $TOOLCHAIN"
echo "  MSRV:           ${MSRV:-<not set>}"
echo "  fmt:            $RUN_FMT"
echo "  lints:          $RUN_LINTS"
echo "  clippy:         $RUN_CLIPPY"
echo "  tests (debug):  $RUN_TESTS"
echo "  build --release:$RUN_RELEASE_BUILD"
echo "  test  --release:$RUN_RELEASE_TEST"
echo

run_ci_for_toolchain() {
  local tc="$1"
  ensure_toolchain_installed "$tc"

  # Best-effort components install
  rustup component add rustfmt --toolchain "$tc" >/dev/null 2>&1 || true
  rustup component add clippy  --toolchain "$tc" >/dev/null 2>&1 || true

  if [[ "$RUN_FMT" -eq 1 ]]; then
    run "cargo fmt ($tc)" cargo +"$tc" fmt --all -- --check
  fi

  if [[ "$RUN_LINTS" -eq 1 ]]; then
    # Auto-discover and run all lint scripts matching lint_*.sh
    shopt -s nullglob
    LINTS=("$ROOT_DIR"/scripts/lint_*.sh)
    shopt -u nullglob

    if [[ ${#LINTS[@]} -gt 0 ]]; then
      for lint in "${LINTS[@]}"; do
        if [[ -x "$lint" ]]; then
          run "lint $(basename "${lint%.sh}")" "$lint"
        else
          warn "Lint script is not executable, skipping: $(basename "$lint")"
        fi
      done
    fi
  fi

  if [[ "$RUN_CLIPPY" -eq 1 ]]; then
    run "cargo clippy ($tc)" cargo +"$tc" "${CLIPPY_ARGS[@]}"
  fi

  if [[ "$RUN_TESTS" -eq 1 ]]; then
    run "cargo test (debug) ($tc)" cargo +"$tc" "${CARGO_TEST_ARGS[@]}"
  fi

  if [[ "$RUN_RELEASE_BUILD" -eq 1 ]]; then
    run "cargo build --release ($tc)" cargo +"$tc" build --workspace --release
  fi

  if [[ "$RUN_RELEASE_TEST" -eq 1 ]]; then
    run "cargo test --release ($tc)" cargo +"$tc" test --workspace --release
  fi
}

run_ci_for_toolchain "$TOOLCHAIN"

if [[ "$RUN_MSRV" -eq 1 ]]; then
  if [[ -z "$MSRV" ]]; then
    warn "--msrv requested but rust-version is not set. Skipping MSRV run."
  else
    echo
    step "Running MSRV suite ($MSRV)"
    run_ci_for_toolchain "$MSRV"
    ok "MSRV suite ($MSRV)"
  fi
fi

echo
ok "All selected CI checks passed."
