//! Local active solve backend boundary.
//!
//! This backend is solver-owned and executes the solver-native runtime pipeline.
//! Keeping this indirection local lets us switch implementations without
//! changing call sites.

use crate::solve_exponential_terms::{
    collect_exp_integer_bases, collect_exp_laurent_terms, collect_exponential_base_terms,
    integer_prime_power, rewrite_exp_bases_to_prime,
};
use crate::{CasError, Simplifier, SolveCtx, SolveStep};
use cas_ast::{substitute_expr_by_id, Constant, Context, Equation, Expr, ExprId, SolutionSet};
// Canonical surd-sign kernel (was a byte-identical private `linear_surd_sign` copy).
use cas_math::root_forms::sign_of_linear_surd as linear_surd_sign;
// Canonical exact surd/nth-root comparators (were private copies; chokepoint A part 2).
use cas_math::root_forms::{cmp_rational_to_nth_root, cmp_rational_to_quadratic_surd};
use cas_solver_core::domain_condition::ImplicitCondition;

use crate::solve_backend_contract::{CoreSolverOptions, SolveBackend};

/// Position of a constant threshold `c` relative to the closed range `[-1, 1]` of `sin`/`cos`,
/// decided EXACTLY over a single quadratic surd (`A + B·√n`, covering rationals and surds). `None`
/// when `c ∈ (-1, 1)` or its position cannot be proven (transcendental / multi-surd) — those are
/// periodic and left to the residual path.
enum TrigThresholdRegion {
    AboveRange,
    BelowRange,
    AtUpperBound,
    AtLowerBound,
}

/// Classification of a candidate root by back-substitution into the original
/// equation, evaluated numerically over the reals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RootCheck {
    /// Both sides evaluate to the same finite real — a genuine solution.
    Verified,
    /// Both sides evaluate to different finite reals — extraneous (drop it).
    Extraneous,
    /// Cannot decide numerically (symbolic/parametric/irrational/undefined) — keep.
    Unknown,
}

/// Structural sign RANGE of a candidate-root expression under the root-filter
/// premise. F10 (frontier-audit 2026-07-14): the constant oracles cannot read a
/// SYMBOLIC-parameter root like `(−√(4a+1)−1)/2`, so the always-negative
/// extraneous root of `√(a−x) = x` leaked unfiltered. This evaluator decides
/// the sign structurally, treating a principal even root (and `|·|`) as `≥ 0`.
///
/// SOUNDNESS SCOPE — root-filter contexts ONLY: the assumption `√u ≥ 0` is
/// valid here because the value being signed is the condition target AT a
/// candidate real solution. For every parameter value where the candidate is a
/// real number, its radicals are real and principal roots are nonnegative; for
/// parameter values where a radicand goes negative, the candidate is not a
/// real number at all and can never be a real solution, so dropping it is
/// vacuously sound too. Do NOT reuse this for definedness decisions
/// (`solve(√x ≥ 0)` must stay `[0, ∞)`, not ℝ — the shared `prove_sign` is
/// deliberately not taught this assumption).
#[derive(Clone, Copy, PartialEq, Debug)]
enum RadicalSignRange {
    Neg,
    NonPos,
    Zero,
    NonNeg,
    Pos,
    Unknown,
}

/// A monotonic function whose inequality result must be intersected with its real
/// argument-domain (which the inversion — square / exponentiate — drops).
#[derive(Clone, Copy)]
enum MonotonicFn {
    /// Even root `√(arg)` / `arg^(1/2k)`: range `[0, ∞)`, domain `{arg ≥ 0}`.
    EvenRoot,
    /// `ln(arg)` / `log(b, arg)`: range `ℝ`, domain `{arg > 0}`.
    Log,
}

/// Split `d` into a single square-root term `±√f` (radicand containing `var`) and
/// the remaining signed terms. Returns `(sign, f, rest_terms)` or None when there
/// is not exactly one such radical.
/// A signed additive term `(sign, expr)` in a decomposition.
type SignedTerm = (i8, ExprId);
/// `(radical_sign, radicand, remaining_signed_terms)` from [`collect_radical_split`].
type RadicalSplit = (i8, ExprId, Vec<SignedTerm>);

/// Local backend facade selected as the active backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalSolveBackend;

impl SolveBackend for LocalSolveBackend {
    fn solve_with_ctx_and_options(
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        opts: CoreSolverOptions,
        ctx: &SolveCtx,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        // Thread the solve value domain into the simplifier's sticky state so
        // solver-internal plain `simplify()` calls fold complex forms
        // (`√(-3) → i·√3`) under ComplexEnabled. Save/restore: the sticky
        // domain must not leak past this solve invocation.
        let previous_domain = simplifier.set_sticky_value_domain(opts.value_domain);
        let result = solve_local_core(eq, var, simplifier, opts, ctx).map(|(set, steps)| {
            // For a NON-STRICT inequality (`f ≤ 0` / `f ≥ 0`) EVERY real root of `f = lhs − rhs` is
            // a solution (`0` satisfies `≤ 0` and `≥ 0`), but the interval sign-analysis drops
            // isolated roots of even-multiplicity factors (`(x−2)²(x+1) ≤ 0` keeps `(−∞,−1]` but
            // loses `{2}`; `x²/(x−1) ≥ 0` keeps `(1,∞)` but loses `{0}`). Union those roots back in
            // — they exclude poles by construction (a pole is not a root of `f = 0`) and are
            // domain-filtered.
            let set = union_non_strict_inequality_roots(eq, var, simplifier, opts, ctx, set);
            (set, steps)
        });
        simplifier.set_sticky_value_domain(previous_domain);
        result
    }
}

/// Position of a `sin`/`cos` RHS `c` relative to the unit interval, decided EXACTLY.
enum TrigUnitClass {
    Zero,       // c = 0
    Unit,       // |c| = 1
    InOpen,     // 0 < |c| < 1
    OutOfRange, // |c| > 1 (no real solution)
}

/// Multiple-angle EXPANSION rewrites (`sin(3x) → 3·sin(x) − 4·sin(x)³`, the
/// quintuple analogue, and the recursive expander) destroy the `trig(n·x) = c`
/// shape inside this handler's own simplifies BEFORE the periodic matcher can
/// see it — the polynomial fallback then asserts a finite arcsin set as the
/// complete answer (family-A soundness bug: `sin(3x)=1/2` lost `2kπ/3`
/// periodicity). Contractions (`Double/Triple Angle Contraction`) stay live —
/// they REBUILD the matchable shape (`sin·cos → sin(2x)/2`).
const MULTIPLE_ANGLE_EXPANSION_RULES: &[&str] = &[
    "Double Angle Identity",
    "Double Angle Expansion",
    "Triple Angle Identity",
    "Quintuple Angle Identity",
    "Recursive Trig Expansion",
];

#[cfg(test)]
mod tests;

mod absolute_value;
mod exponential_log;
mod general;
mod inequalities;
mod polynomial;
mod radicals;
mod rational;
mod sign_domain;
mod support;
mod trigonometric;

use absolute_value::*;
use exponential_log::*;
use general::*;
pub(crate) use inequalities::*;
use polynomial::*;
use radicals::*;
use rational::*;
use sign_domain::*;
use support::*;
pub(crate) use trigonometric::*;
