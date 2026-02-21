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
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_solver_core::solution_set::sort_and_dedup_exprs;
use num_rational::BigRational;
use num_traits::Zero;

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
        // Only handle equality
        if eq.op != RelOp::Eq {
            return None;
        }

        // Move everything to LHS: lhs - rhs = 0
        let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
        let (sim_diff, _) = simplifier.simplify(diff);

        // Expand to canonical polynomial form
        let expanded = crate::expand::expand(&mut simplifier.context, sim_diff);

        // Extract polynomial coefficients: [a0, a1, ..., an] where poly = a0 + a1*x + ... + an*x^n
        let coeffs = cas_solver_core::rational_roots::extract_poly_coefficients(
            &mut simplifier.context,
            expanded,
            var,
            MAX_DEGREE,
        )?;

        let degree = coeffs.len() - 1;

        // Only handle degree ≥ 3 (degree ≤ 2 is handled by QuadraticStrategy/Linear)
        if !(3..=MAX_DEGREE).contains(&degree) {
            return None;
        }

        // All coefficients must be numeric (rational)
        let rat_coeffs: Vec<BigRational> = coeffs
            .iter()
            .map(|&c| cas_solver_core::rational_roots::get_rational(&simplifier.context, c))
            .collect::<Option<Vec<_>>>()?;

        // All zeros check
        if rat_coeffs.iter().all(|c| c.is_zero()) {
            return Some(Ok((SolutionSet::AllReals, vec![])));
        }

        // Find rational roots (pure deflation in solver core)
        let mut roots = cas_solver_core::rational_roots::extract_candidate_roots(
            &mut simplifier.context,
            rat_coeffs,
            MAX_CANDIDATES,
        )
        .into_iter()
        .map(|root_expr| {
            let (sim_root, _) = simplifier.simplify(root_expr);
            sim_root
        })
        .collect::<Vec<_>>();

        if roots.is_empty() {
            return None; // No roots found, let other strategies try
        }

        // Dedup roots
        sort_and_dedup_exprs(&simplifier.context, &mut roots);

        let steps = if simplifier.collect_steps() {
            vec![SolveStep {
                description: format!(
                    "Applied Rational Root Theorem to degree-{} polynomial",
                    degree
                ),
                equation_after: Equation {
                    lhs: expanded,
                    rhs: simplifier.context.num(0),
                    op: RelOp::Eq,
                },
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            }]
        } else {
            vec![]
        };

        Some(Ok((SolutionSet::Discrete(roots), steps)))
    }

    fn should_verify(&self) -> bool {
        true // Verify roots against original equation
    }
}
