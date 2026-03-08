//! Hold semantics and expression substitution utilities.
//!
//! Functions for managing HoldAll function semantics, unwrapping hold barriers,
//! and performing structural expression substitution by ExprId.

use cas_ast::{Context, ExprId};

// =============================================================================
// HoldAll function semantics
// =============================================================================

/// Returns true if a function has HoldAll semantics, meaning its arguments
/// should NOT be simplified before the function rule is applied.
/// This is crucial for functions like poly_gcd that need to see the raw
/// multiplicative structure of their arguments.
/// Also includes hold function which is an internal invisible barrier.
pub(super) fn is_hold_all_function(name: &str) -> bool {
    matches!(name, "poly_gcd" | "pgcd") || cas_ast::hold::is_hold_name(name)
}

/// Unwrap top-level __hold() wrapper after simplification.
/// This is called at the end of eval/simplify so the user sees clean results
/// without the INTERNAL barrier visible (user-facing hold() is preserved).
pub(super) fn unwrap_hold_top(ctx: &Context, expr: ExprId) -> ExprId {
    cas_ast::hold::unwrap_internal_hold(ctx, expr)
}
