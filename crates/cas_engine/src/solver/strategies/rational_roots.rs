//! RationalRootsStrategy — solves polynomial equations of degree ≥ 3
//! with all-numeric (rational) coefficients using the Rational Root Theorem
//! plus synthetic division (deflation).
//!
//! Pipeline:
//! 1. Extract univariate polynomial coefficients from `simplify(lhs - rhs)`
//! 2. Normalize to integer coefficients (scale by LCM of denominators)
//! 3. Enumerate candidate rational roots ±p/q
//! 4. Verify each candidate via exact Horner evaluation
//! 5. Deflate by confirmed roots (synthetic division)
//! 6. Delegate residual polynomial (degree ≤ 2) to existing strategies

use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{medium_step, SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, SolutionSet};
use cas_solver_core::rational_roots::{
    extract_poly_coefficients, plan_rational_roots_strategy_step_with_zero_rhs,
    solve_numeric_coeff_polynomial,
    solve_rational_roots_strategy_result_for_equation_with_and_item_with_state,
};
use cas_solver_core::solution_set::sort_and_dedup_exprs;

/// Maximum number of candidate rational roots to try before bailing.
/// Prevents combinatorial blowup on polynomials with large leading/constant coefficients.
const MAX_CANDIDATES: usize = 200;

/// Maximum polynomial degree we attempt.
const MAX_DEGREE: usize = 10;

pub struct RationalRootsStrategy;

impl SolverStrategy for RationalRootsStrategy {
    fn name(&self) -> &str {
        "Rational Roots"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        _opts: &SolverOptions,
        _ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let include_item = simplifier.collect_steps();
        let zero_rhs = simplifier.context.num(0);
        let solved = solve_rational_roots_strategy_result_for_equation_with_and_item_with_state(
            simplifier,
            eq,
            var,
            3,
            MAX_DEGREE,
            MAX_CANDIDATES,
            include_item,
            |simplifier, lhs, rhs| simplifier.context.add(Expr::Sub(lhs, rhs)),
            |simplifier, expr| simplifier.simplify(expr).0,
            |simplifier, expr| crate::expand::expand(&mut simplifier.context, expr),
            |simplifier, expanded, var_name, max_degree| {
                extract_poly_coefficients(&mut simplifier.context, expanded, var_name, max_degree)
            },
            |simplifier, coeffs, min_degree, max_degree, max_candidates| {
                solve_numeric_coeff_polynomial(
                    &mut simplifier.context,
                    coeffs,
                    min_degree,
                    max_degree,
                    max_candidates,
                )
            },
            |simplifier, roots| {
                sort_and_dedup_exprs(&simplifier.context, roots);
            },
            |_simplifier, expanded, degree| {
                plan_rational_roots_strategy_step_with_zero_rhs(expanded, degree, zero_rhs)
            },
            |item| medium_step(item.description().to_string(), item.equation),
        )?;
        Some(Ok(solved))
    }

    fn should_verify(&self) -> bool {
        true // Verify roots against original equation
    }
}
