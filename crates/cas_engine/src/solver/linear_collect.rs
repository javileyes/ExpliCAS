//! Linear Collect Strategy for solving equations with additive terms.
//!
//! This module handles equations where the target variable appears in multiple
//! additive terms, like A = P + P*r*t. It factors out the variable and solves
//! by division, returning a Conditional solution when the coefficient might be zero.
//!
//! Example: A = P + P*r*t
//! 1. Move all to LHS: P + P*r*t - A = 0
//! 2. Factor: P*(1 + r*t) - A = 0
//! 3. Solve: P = A / (1 + r*t)  [guard: 1 + r*t ≠ 0]

use cas_ast::{Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::linear_solution::{build_linear_solution_set, NonZeroStatus};
use cas_solver_core::linear_terms::{build_sum, split_linear_term, TermClass};

use crate::engine::Simplifier;
use crate::nary::{add_terms_signed, Sign};
use crate::solver::isolation::contains_var;
use crate::solver::SolveStep;

fn proof_to_nonzero_status(proof: crate::domain::Proof) -> NonZeroStatus {
    match proof {
        crate::domain::Proof::Proven | crate::domain::Proof::ProvenImplicit => {
            NonZeroStatus::NonZero
        }
        crate::domain::Proof::Disproven => NonZeroStatus::Zero,
        crate::domain::Proof::Unknown => NonZeroStatus::Unknown,
    }
}

/// Try to solve a linear equation where variable appears in multiple additive terms.
///
/// Returns Some((SolutionSet, steps)) if successful, None if not applicable.
///
/// Example: P + P*r*t - A = 0 → P = A / (1 + r*t) with guard 1+r*t ≠ 0
pub(crate) fn try_linear_collect(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<(SolutionSet, Vec<SolveStep>)> {
    let ctx = &mut simplifier.context;

    // 1. Build expr = lhs - rhs (move everything to LHS, so expr = 0)
    let expr = ctx.add(Expr::Sub(lhs, rhs));
    let (expr, _) = simplifier.simplify(expr);

    // 2. Flatten as sum of SIGNED terms using canonical utility
    let terms = add_terms_signed(&simplifier.context, expr);

    // 3. Classify each term, respecting signs
    let mut coeff_parts: Vec<ExprId> = Vec::new();
    let mut const_parts: Vec<ExprId> = Vec::new();

    for (term, sign) in terms {
        match split_linear_term(&mut simplifier.context, term, var) {
            TermClass::Const(_) => {
                // Apply sign to constant term
                let signed_term = match sign {
                    Sign::Pos => term,
                    Sign::Neg => simplifier.context.add(Expr::Neg(term)),
                };
                const_parts.push(signed_term);
            }
            TermClass::Linear(c) => {
                // Convert None (implicit 1) to explicit 1
                let coef = c.unwrap_or_else(|| simplifier.context.num(1));
                // Apply sign to coefficient
                let signed_coef = match sign {
                    Sign::Pos => coef,
                    Sign::Neg => simplifier.context.add(Expr::Neg(coef)),
                };
                coeff_parts.push(signed_coef);
            }
            TermClass::NonLinear => {
                // Variable appears non-linearly, this strategy doesn't apply
                return None;
            }
        }
    }

    // If no linear terms found, strategy doesn't apply
    if coeff_parts.is_empty() {
        return None;
    }

    // 4. Build coeff = sum of linear coefficients
    let coeff = build_sum(&mut simplifier.context, &coeff_parts);
    let (coeff, _) = simplifier.simplify(coeff);

    // 5. Build const = sum of constant parts (with sign flipped for solution)
    // coeff*var + const = 0 → var = -const / coeff
    let const_sum = build_sum(&mut simplifier.context, &const_parts);
    let neg_const = simplifier.context.add(Expr::Neg(const_sum));
    let (neg_const, _) = simplifier.simplify(neg_const);

    // 6. Build solution: var = -const / coeff
    let solution = simplifier.context.add(Expr::Div(neg_const, coeff));
    let (solution, _) = simplifier.simplify(solution);

    // 7. Build step description
    let mut steps = Vec::new();
    if simplifier.collect_steps() {
        let var_expr = simplifier.context.var(var);
        steps.push(SolveStep {
            description: format!(
                "Collect terms in {} and factor: {} · {} = {}",
                var,
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: coeff
                },
                var,
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: neg_const
                }
            ),
            equation_after: cas_ast::Equation {
                lhs: simplifier.context.add(Expr::Mul(coeff, var_expr)),
                rhs: neg_const,
                op: RelOp::Eq,
            },
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
        steps.push(SolveStep {
            description: format!(
                "Divide both sides by {}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: coeff
                }
            ),
            equation_after: cas_ast::Equation {
                lhs: var_expr,
                rhs: solution,
                op: RelOp::Eq,
            },
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    // 8. Derive proof statuses for coefficient/constant degeneracy checks.
    // Keep previous behavior: only attempt proof when coefficient is var-free.
    let mut coef_status = NonZeroStatus::Unknown;
    let mut constant_status = NonZeroStatus::Unknown;

    if !contains_var(&simplifier.context, coeff, var) {
        use crate::helpers::prove_nonzero;
        coef_status = proof_to_nonzero_status(prove_nonzero(&simplifier.context, coeff));
        if coef_status == NonZeroStatus::Zero {
            constant_status =
                proof_to_nonzero_status(prove_nonzero(&simplifier.context, neg_const));
        }
    }

    let solution_set =
        build_linear_solution_set(coeff, neg_const, solution, coef_status, constant_status);

    Some((solution_set, steps))
}

/// Try to solve using the structural linear form extractor.
///
/// This is an alternative to the term-based approach that works
/// better for expressions like `y*(1+x)` where the coefficient
/// is itself an expression.
pub(crate) fn try_linear_collect_v2(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<(SolutionSet, Vec<SolveStep>)> {
    let kernel = cas_solver_core::linear_kernel::derive_linear_solve_kernel(
        &mut simplifier.context,
        lhs,
        rhs,
        var,
    )?;

    // Simplify for cleaner display
    let (coef, _) = simplifier.simplify(kernel.coef);
    let (constant, _) = simplifier.simplify(kernel.constant);

    // Solution: var = -constant / coef  (from coef*var + constant = 0)
    let neg_constant = simplifier.context.add(Expr::Neg(constant));
    let solution = simplifier.context.add(Expr::Div(neg_constant, coef));
    let (solution, _) = simplifier.simplify(solution);

    // Build steps
    let mut steps = Vec::new();

    if simplifier.collect_steps() {
        // Step 1: Show the factored form
        let var_id = simplifier.context.var(var);
        let coef_times_var = simplifier.context.add(Expr::Mul(coef, var_id));
        let factored_lhs = simplifier.context.add(Expr::Add(coef_times_var, constant));
        let zero = simplifier.context.num(0);

        steps.push(SolveStep {
            description: format!("Collect terms in {}", var),
            equation_after: cas_ast::Equation {
                lhs: factored_lhs,
                rhs: zero,
                op: RelOp::Eq,
            },
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });

        // Step 2: Divide by coefficient
        steps.push(SolveStep {
            description: format!(
                "Divide by {}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: coef
                }
            ),
            equation_after: cas_ast::Equation {
                lhs: var_id,
                rhs: solution,
                op: RelOp::Eq,
            },
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    // Derive proof statuses for coefficient/constant degeneracy checks.
    // Keep previous behavior: only attempt proof when coefficient is var-free.
    let mut coef_status = NonZeroStatus::Unknown;
    let mut constant_status = NonZeroStatus::Unknown;

    if !contains_var(&simplifier.context, coef, var) {
        use crate::helpers::prove_nonzero;
        coef_status = proof_to_nonzero_status(prove_nonzero(&simplifier.context, coef));
        if coef_status == NonZeroStatus::Zero {
            constant_status = proof_to_nonzero_status(prove_nonzero(&simplifier.context, constant));
        }
    }

    let solution_set =
        build_linear_solution_set(coef, constant, solution, coef_status, constant_status);

    Some((solution_set, steps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn test_add_terms_signed() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");

        // a + b + c
        let ab = ctx.add(Expr::Add(a, b));
        let abc = ctx.add(Expr::Add(ab, c));

        let terms = add_terms_signed(&ctx, abc);
        assert_eq!(terms.len(), 3);
    }

    #[test]
    fn test_split_linear_term_const() {
        let mut ctx = Context::new();
        let a = ctx.var("A");

        match split_linear_term(&mut ctx, a, "P") {
            TermClass::Const(_) => {}
            _ => panic!("A should be Const with respect to P"),
        }
    }

    #[test]
    fn test_split_linear_term_var() {
        let mut ctx = Context::new();
        let p = ctx.var("P");

        match split_linear_term(&mut ctx, p, "P") {
            TermClass::Linear(_) => {}
            _ => panic!("P should be Linear(1) with respect to P"),
        }
    }
}
