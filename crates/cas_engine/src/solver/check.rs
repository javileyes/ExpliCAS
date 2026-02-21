//! Solution verification module.
//!
//! Verifies solver solutions by substituting back into the original equation
//! and checking if the result simplifies to zero.
//!
//! Uses a **2-phase** approach:
//! - **Phase 1 (Strict):** Domain-honest simplification that won't erase
//!   conditions (e.g., won't cancel `x/x → 1` without proving `x ≠ 0`).
//! - **Phase 2 (Generic fallback):** Only fires when the Strict residual is
//!   **variable-free** (ground check). This safely handles cases where
//!   `has_undefined_risk` blocks cancellation of concrete constants like
//!   `sqrt(2)/sqrt(2)`, without risking parametric domain erasure.
//!
//! # Example
//!
//! ```ignore
//! let status = verify_solution(simplifier, &equation, "x", solution);
//! match status {
//!     VerifyStatus::Verified => println!("✓ verified"),
//!     VerifyStatus::Unverifiable { reason, .. } => println!("⚠ {}", reason),
//!     VerifyStatus::NotCheckable { reason } => println!("ℹ {}", reason),
//! }
//! ```

use cas_ast::{Equation, ExprId, SolutionSet};
use cas_formatter::DisplayExpr;
use cas_math::expr_predicates::contains_variable;
use cas_solver_core::isolation_utils::is_numeric_zero;

use crate::engine::Simplifier;
pub use cas_solver_core::verification::{VerifyResult, VerifyStatus, VerifySummary};

/// Verify a single solution by substituting into the equation.
///
/// # Algorithm
/// 1. Substitute `var := solution` in both `lhs` and `rhs`
/// 2. Compute `diff = lhs_sub - rhs_sub`
/// 3. **Phase 1:** Simplify with Strict mode (domain-honest)
/// 4. If result is `0` → `Verified`
/// 5. **Phase 1.5:** Fold numeric islands (constant subtrees → numbers),
///    then re-simplify with Strict. This handles expressions like `sqrt(2)/sqrt(2)`
///    without switching to Generic mode.
/// 6. **Phase 2:** If residual is still variable-free, retry with Generic mode
/// 7. If result is `0` → `Verified`, otherwise → `Unverifiable`
pub fn verify_solution(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solution: ExprId,
) -> VerifyStatus {
    // Step 1: Substitute solution and build residual diff = lhs_sub - rhs_sub
    let diff = cas_solver_core::verify_substitution::substitute_equation_diff(
        &mut simplifier.context,
        equation,
        var,
        solution,
    );

    // Phase 1: Strict mode — domain-honest, won't erase conditions
    let strict_opts = crate::SimplifyOptions {
        shared: crate::phase::SharedSemanticConfig {
            semantics: crate::semantics::EvalConfig {
                domain_mode: crate::domain::DomainMode::Strict,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let (strict_result, _, _) = simplifier.simplify_with_stats(diff, strict_opts.clone());

    // Check if Strict already gives us 0
    if is_numeric_zero(&simplifier.context, strict_result) {
        return VerifyStatus::Verified;
    }

    // Phase 1.5: Numeric island folding — fold constant subtrees to numbers,
    // then re-simplify with Strict.  This lets expressions like `sqrt(2)/sqrt(2)`
    // reduce without leaving Strict mode.
    if contains_variable(&simplifier.context, strict_result) {
        cas_solver_core::verify_stats::record_attempted();
        let folded =
            super::numeric_islands::fold_numeric_islands(&mut simplifier.context, strict_result);
        if folded != strict_result {
            cas_solver_core::verify_stats::record_changed();
            let (folded_result, _, _) = simplifier.simplify_with_stats(folded, strict_opts);
            if is_numeric_zero(&simplifier.context, folded_result) {
                cas_solver_core::verify_stats::record_verified();
                return VerifyStatus::Verified;
            }
        }
    }

    // Phase 2: Generic fallback — ONLY when residual is variable-free.
    // If no variables remain, this is a ground check (concrete values only),
    // so Generic mode can't erase parametric domain conditions — there are none.
    // This handles cases where Strict blocks cancellation of constants like
    // sqrt(2)/sqrt(2) because prove_nonzero doesn't fully evaluate them.
    if !contains_variable(&simplifier.context, strict_result) {
        let generic_opts = crate::SimplifyOptions {
            shared: crate::phase::SharedSemanticConfig {
                semantics: crate::semantics::EvalConfig {
                    domain_mode: crate::domain::DomainMode::Generic,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let (generic_result, _, _) = simplifier.simplify_with_stats(diff, generic_opts);

        if is_numeric_zero(&simplifier.context, generic_result) {
            return VerifyStatus::Verified;
        }
    }

    // Neither phase verified — report the Strict residual
    let residual_str = DisplayExpr {
        context: &simplifier.context,
        id: strict_result,
    }
    .to_string();
    VerifyStatus::Unverifiable {
        residual: strict_result,
        reason: format!("residual: {}", residual_str),
    }
}

/// Verify a solution set, handling all SolutionSet variants.
pub fn verify_solution_set(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solutions: &SolutionSet,
) -> VerifyResult {
    let mut verify_one = |sol: ExprId| verify_solution(simplifier, equation, var, sol);
    cas_solver_core::verification::verify_solution_set_with(solutions, &mut verify_one)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;
    use cas_ast::RelOp;

    fn make_simplifier() -> Simplifier {
        Simplifier::with_default_rules()
    }

    #[test]
    fn test_verify_linear_solution() {
        // x + 2 = 5, solution x = 3
        let mut s = make_simplifier();
        let x = s.context.var("x");
        let two = s.context.num(2);
        let five = s.context.num(5);
        let three = s.context.num(3);

        let lhs = s.context.add(Expr::Add(x, two));
        let eq = Equation {
            lhs,
            rhs: five,
            op: RelOp::Eq,
        };

        let status = verify_solution(&mut s, &eq, "x", three);
        assert!(matches!(status, VerifyStatus::Verified));
    }

    #[test]
    fn test_verify_quadratic_solutions() {
        // x^2 = 4, solutions x = 2 and x = -2
        let mut s = make_simplifier();
        let x = s.context.var("x");
        let two = s.context.num(2);
        let four = s.context.num(4);
        let neg_two = s.context.num(-2);

        let lhs = s.context.add(Expr::Pow(x, two));
        let eq = Equation {
            lhs,
            rhs: four,
            op: RelOp::Eq,
        };

        // x = 2 should verify
        let status1 = verify_solution(&mut s, &eq, "x", two);
        assert!(matches!(status1, VerifyStatus::Verified));

        // x = -2 should verify
        let status2 = verify_solution(&mut s, &eq, "x", neg_two);
        assert!(matches!(status2, VerifyStatus::Verified));
    }

    #[test]
    fn test_verify_wrong_solution() {
        // x + 2 = 5, wrong solution x = 4
        let mut s = make_simplifier();
        let x = s.context.var("x");
        let two = s.context.num(2);
        let five = s.context.num(5);
        let four = s.context.num(4);

        let lhs = s.context.add(Expr::Add(x, two));
        let eq = Equation {
            lhs,
            rhs: five,
            op: RelOp::Eq,
        };

        let status = verify_solution(&mut s, &eq, "x", four);
        assert!(matches!(status, VerifyStatus::Unverifiable { .. }));
    }

    #[test]
    fn test_verify_solution_set_discrete() {
        // x^2 = 4, solutions {2, -2}
        let mut s = make_simplifier();
        let x = s.context.var("x");
        let two = s.context.num(2);
        let four = s.context.num(4);
        let neg_two = s.context.num(-2);

        let lhs = s.context.add(Expr::Pow(x, two));
        let eq = Equation {
            lhs,
            rhs: four,
            op: RelOp::Eq,
        };

        let solutions = SolutionSet::Discrete(vec![two, neg_two]);
        let result = verify_solution_set(&mut s, &eq, "x", &solutions);

        assert_eq!(result.summary, VerifySummary::AllVerified);
        assert_eq!(result.solutions.len(), 2);
    }

    #[test]
    fn test_verify_all_reals_not_checkable() {
        let mut s = make_simplifier();
        let one = s.context.num(1);
        let eq = Equation {
            lhs: one,
            rhs: one,
            op: RelOp::Eq,
        };

        let solutions = SolutionSet::AllReals;
        let result = verify_solution_set(&mut s, &eq, "x", &solutions);

        assert_eq!(result.summary, VerifySummary::NotCheckable);
    }
}
