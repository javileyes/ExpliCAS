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
    solve_numeric_coeff_polynomial, solve_rational_roots_step_pipeline_with_item,
    NumericPolynomialSolveOutcome,
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
        if eq.op != cas_ast::RelOp::Eq {
            return None;
        }

        let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
        let diff = simplifier.simplify(diff).0;
        let expanded = crate::expand::expand(&mut simplifier.context, diff);
        let coeffs = extract_poly_coefficients(&mut simplifier.context, expanded, var, MAX_DEGREE)?;
        let outcome = solve_numeric_coeff_polynomial(
            &mut simplifier.context,
            &coeffs,
            3,
            MAX_DEGREE,
            MAX_CANDIDATES,
        )?;
        let (degree, mut roots) = match outcome {
            NumericPolynomialSolveOutcome::AllReals => {
                return Some(Ok((SolutionSet::AllReals, vec![])));
            }
            NumericPolynomialSolveOutcome::CandidateRoots { degree, roots } => (
                degree,
                roots
                    .into_iter()
                    .map(|expr| simplifier.simplify(expr).0)
                    .collect::<Vec<_>>(),
            ),
        };
        if roots.is_empty() {
            return None;
        }
        sort_and_dedup_exprs(&simplifier.context, &mut roots);

        let step = plan_rational_roots_strategy_step_with_zero_rhs(expanded, degree, zero_rhs);
        let steps = solve_rational_roots_step_pipeline_with_item(step, include_item, |item| {
            medium_step(item.description().to_string(), item.equation)
        });
        Some(Ok((SolutionSet::Discrete(roots), steps)))
    }

    fn should_verify(&self) -> bool {
        true // Verify roots against original equation
    }
}
