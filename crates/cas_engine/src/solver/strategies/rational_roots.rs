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
    extract_poly_coefficients, plan_rational_roots_step, solve_numeric_coeff_polynomial,
    solve_rational_roots_strategy_with_and_item, RationalRootsExecutionItem,
};
use std::cell::RefCell;

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
        let simplifier_cell = RefCell::new(simplifier);
        let solved = solve_rational_roots_strategy_with_and_item(
            eq.lhs,
            eq.rhs,
            eq.op.clone(),
            var,
            3,
            MAX_DEGREE,
            MAX_CANDIDATES,
            include_item,
            |left, right| {
                let mut s_ref = simplifier_cell.borrow_mut();
                s_ref.context.add(Expr::Sub(left, right))
            },
            |expr| {
                let mut s_ref = simplifier_cell.borrow_mut();
                let (simplified, _) = s_ref.simplify(expr);
                simplified
            },
            |expr| {
                let mut s_ref = simplifier_cell.borrow_mut();
                crate::expand::expand(&mut s_ref.context, expr)
            },
            |expanded, name, max_degree| {
                let mut s_ref = simplifier_cell.borrow_mut();
                extract_poly_coefficients(&mut s_ref.context, expanded, name, max_degree)
            },
            |coeffs, min_degree, max_degree, max_candidates| {
                let mut s_ref = simplifier_cell.borrow_mut();
                solve_numeric_coeff_polynomial(
                    &mut s_ref.context,
                    coeffs,
                    min_degree,
                    max_degree,
                    max_candidates,
                )
            },
            |roots| {
                let mut s_ref = simplifier_cell.borrow_mut();
                cas_solver_core::solution_set::sort_and_dedup_exprs(&mut s_ref.context, roots);
            },
            |expanded, degree| {
                let mut s_ref = simplifier_cell.borrow_mut();
                plan_rational_roots_step(&mut s_ref.context, expanded, degree)
            },
            |item: RationalRootsExecutionItem| {
                medium_step(item.description().to_string(), item.equation)
            },
        )?;
        Some(Ok((solved.solution_set, solved.steps)))
    }

    fn should_verify(&self) -> bool {
        true // Verify roots against original equation
    }
}
