//! Public solve entry points.
//!
//! Keeps the externally visible API (`solve*`) separate from `solve_inner`
//! orchestration internals.

use crate::engine::Simplifier;
use crate::error::CasError;
use cas_ast::SolutionSet;
use cas_solver_core::solve_types::cleanup_display_solve_steps;

use super::solve_core::solve_inner;
use super::{DisplaySolveSteps, SolveDiagnostics, SolveStep, SolverOptions};

/// Solve with default options (for backward compatibility with tests).
/// Uses RealOnly domain and Generic mode.
///
/// This creates a fresh `SolveCtx`; conditions are NOT propagated
/// to any parent context. For recursive calls from strategies that
/// need to accumulate conditions, use [`solve_with_ctx_and_options`] instead.
pub fn solve(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    solve_with_options(eq, var, simplifier, SolverOptions::default())
}

/// V2.9.8: Solve with type-enforced display-ready steps.
///
/// This is the PREFERRED entry point for display-facing code (REPL, timeline, JSON API).
/// Returns `DisplaySolveSteps` which enforces that all renderers consume post-processed
/// steps, eliminating bifurcation between text/timeline outputs at compile time.
///
/// The cleanup is applied automatically based on `opts.detailed_steps`:
/// - `true` -> 5 atomic sub-steps for Normal/Verbose verbosity
/// - `false` -> 3 compact steps for Succinct verbosity
pub fn solve_with_display_steps(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, DisplaySolveSteps, SolveDiagnostics), CasError> {
    // Create a SolveCtx with a fresh accumulator — all recursive calls
    // through solve_with_ctx_and_options will push conditions into this shared set.
    let ctx = super::SolveCtx::default();
    let result = solve_inner(eq, var, simplifier, opts, &ctx);
    cas_solver_core::solve_types::finalize_display_solve_with_ctx(
        &ctx,
        result,
        crate::collect_assumption_records,
        |raw_steps| {
            cleanup_display_solve_steps(
                &mut simplifier.context,
                raw_steps,
                opts.detailed_steps,
                var,
            )
        },
    )
}

/// Solve with options but no shared context.
///
/// Creates a fresh, isolated `SolveCtx`. Conditions derived here do NOT
/// propagate to any parent context. Prefer [`solve_with_ctx_and_options`] inside
/// strategies that already hold a `&SolveCtx`.
pub(crate) fn solve_with_options(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let ctx = super::SolveCtx::default();
    solve_inner(eq, var, simplifier, opts, &ctx)
}

// NOTE: Pre-solve exponent normalization (Div(p,q) -> Number(p/q)) is handled
// by canonicalization rules:
//   - rules/rational_canonicalization.rs
// Common additive cancellation remains an equation-level solver primitive and
// lives under:
//   - solver/cancel_common_terms.rs
// (with shared kernels in cas_solver_core::cancel_common_terms).

/// Solve with a shared `SolveCtx` and explicit options.
///
/// This should be used by recursive strategy/isolation paths so nested solves
/// preserve semantic/domain options from the top-level invocation.
pub fn solve_with_ctx_and_options(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    solve_inner(eq, var, simplifier, opts, ctx)
}
