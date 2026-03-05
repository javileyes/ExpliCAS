//! Shared conservative phase-budget presets.
//!
//! These values are intentionally small and are used in bounded helper flows
//! (ground non-zero probing and numeric-island folding) to avoid expensive
//! rewrites and recursion loops.

/// Core phase iterations for conservative probing.
pub const CONSERVATIVE_CORE_ITERS: usize = 4;
/// Transform phase iterations for conservative probing.
pub const CONSERVATIVE_TRANSFORM_ITERS: usize = 2;
/// Rationalize phase iterations for conservative probing.
pub const CONSERVATIVE_RATIONALIZE_ITERS: usize = 0;
/// Post phase iterations for conservative probing.
pub const CONSERVATIVE_POST_ITERS: usize = 2;
/// Total rewrite cap for conservative probing.
pub const CONSERVATIVE_MAX_TOTAL_REWRITES: usize = 50;
