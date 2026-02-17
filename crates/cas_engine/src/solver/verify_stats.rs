//! Phase 1.5 instrumentation counters for `verify_solution`.
//!
//! Gated behind `VERIFY_STATS=1`.  Counters are global atomics so they work
//! correctly across parallel `#[test]` threads.
//!
//! Usage:
//! ```ignore
//! verify_stats::reset_stats();   // zeros counters + refreshes env
//! // … run benchmark …
//! verify_stats::dump_stats();    // prints summary if enabled
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

// ---------------------------------------------------------------------------
// Enable flag
// ---------------------------------------------------------------------------

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Re-read `VERIFY_STATS` from the environment and update `ENABLED`.
fn refresh_enabled_from_env() {
    let on = std::env::var("VERIFY_STATS")
        .map(|v| v == "1")
        .unwrap_or(false);
    ENABLED.store(on, Relaxed);
}

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------

/// Number of times Phase 1.5 was entered (residual contained variables and
/// `fold_numeric_islands` was actually called).
static PHASE15_ATTEMPTED: AtomicUsize = AtomicUsize::new(0);

/// Number of times island folding produced a structurally different tree
/// (`folded != strict_result`).
static PHASE15_CHANGED: AtomicUsize = AtomicUsize::new(0);

/// Number of times re-Strict simplification on the folded tree yielded `0`,
/// completing verification.
static PHASE15_VERIFIED: AtomicUsize = AtomicUsize::new(0);

/// Number of individual ground islands rejected by the size/depth guard in
/// `try_fold_island`.  This is a **per-island** counter, not per-verify-call,
/// so it may exceed `phase15_attempted`.
static PHASE15_SKIPPED_LIMITS: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// Recording (hot path — 1 atomic load + conditional fetch_add)
// ---------------------------------------------------------------------------

pub(crate) fn record_attempted() {
    if ENABLED.load(Relaxed) {
        PHASE15_ATTEMPTED.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_changed() {
    if ENABLED.load(Relaxed) {
        PHASE15_CHANGED.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_verified() {
    if ENABLED.load(Relaxed) {
        PHASE15_VERIFIED.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_skipped_limits() {
    if ENABLED.load(Relaxed) {
        PHASE15_SKIPPED_LIMITS.fetch_add(1, Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Zero all counters and refresh the enable flag from the environment.
/// Call at the start of a benchmark run.
pub fn reset_stats() {
    refresh_enabled_from_env();
    PHASE15_ATTEMPTED.store(0, Relaxed);
    PHASE15_CHANGED.store(0, Relaxed);
    PHASE15_VERIFIED.store(0, Relaxed);
    PHASE15_SKIPPED_LIMITS.store(0, Relaxed);
}

/// Print a one-line summary to stderr if stats collection is enabled.
/// Call at the end of a benchmark run.
pub fn dump_stats() {
    refresh_enabled_from_env();
    if !ENABLED.load(Relaxed) {
        return;
    }
    eprintln!(
        "Phase1.5 stats: attempted={}, changed={}, verified={}, skipped_limits={}",
        PHASE15_ATTEMPTED.load(Relaxed),
        PHASE15_CHANGED.load(Relaxed),
        PHASE15_VERIFIED.load(Relaxed),
        PHASE15_SKIPPED_LIMITS.load(Relaxed),
    );
}
