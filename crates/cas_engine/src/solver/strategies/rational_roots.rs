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

        // Find rational roots
        let mut roots = Vec::new();
        let mut current_coeffs = rat_coeffs;

        loop {
            if current_coeffs.len() <= 1 {
                break;
            }

            // Strip trailing zeros (factor out x): each trailing zero = root at x=0
            while current_coeffs.len() > 1 && current_coeffs[0].is_zero() {
                current_coeffs.remove(0);
                let zero_expr = simplifier.context.num(0);
                roots.push(zero_expr);
            }

            if current_coeffs.len() <= 1 {
                break;
            }

            let deg = current_coeffs.len() - 1;
            if deg <= 2 {
                // Delegate to quadratic/linear
                break;
            }

            // Normalize to integer coefficients
            let int_coeffs =
                cas_solver_core::rational_roots::normalize_to_integers(&current_coeffs);

            // Generate candidates
            let candidates = cas_solver_core::rational_roots::rational_root_candidates(
                &int_coeffs,
                MAX_CANDIDATES,
            );
            if candidates.is_empty() {
                break;
            }

            // Try each candidate
            let mut found_root = false;
            for candidate in &candidates {
                if cas_solver_core::rational_roots::horner_eval(&current_coeffs, candidate)
                    .is_zero()
                {
                    // Confirmed root! Add to results
                    let root_expr = cas_solver_core::rational_roots::rational_to_expr(
                        &mut simplifier.context,
                        candidate,
                    );
                    let (sim_root, _) = simplifier.simplify(root_expr);
                    roots.push(sim_root);

                    // Deflate
                    current_coeffs = cas_solver_core::rational_roots::synthetic_division(
                        &current_coeffs,
                        candidate,
                    );
                    found_root = true;
                    break; // restart candidate search on deflated polynomial
                }
            }

            if !found_root {
                break; // no rational roots found in remaining polynomial
            }
        }

        // Handle residual polynomial (degree ≤ 2) via solver core.
        if current_coeffs.len() == 3 || current_coeffs.len() == 2 {
            for root_expr in cas_solver_core::rational_roots::solve_residual_degree_leq_two(
                &mut simplifier.context,
                &current_coeffs,
            ) {
                let (sim_root, _) = simplifier.simplify(root_expr);
                roots.push(sim_root);
            }
        }
        // else: degree ≥ 3 with no rational roots — can't solve further, but we may have partial roots

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
