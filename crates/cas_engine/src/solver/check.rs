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
use cas_math::expr_predicates::contains_variable;
use cas_solver_core::isolation_utils::is_numeric_zero;
use cas_solver_core::solution_check::{
    verify_solution_with_runtime as verify_solution_with_core_runtime, SolutionCheckRuntime,
};
use cas_solver_core::verification::{verify_solution_set_with_runtime, VerifySolutionSetRuntime};
use cas_solver_core::verify_substitution::{
    verify_solution_with_runtime as verify_solution_with_equivalence_core_runtime,
    VerifySolutionRuntime,
};

use crate::engine::Simplifier;
use crate::solver::render_expr as solver_render_expr;
pub use cas_solver_core::verification::{VerifyResult, VerifyStatus, VerifySummary};

struct EngineSolutionCheckRuntime<'a> {
    simplifier: &'a mut Simplifier,
}

impl SolutionCheckRuntime for EngineSolutionCheckRuntime<'_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn simplify_strict(&mut self, expr: ExprId) -> ExprId {
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
        self.simplifier.simplify_with_stats(expr, strict_opts).0
    }

    fn simplify_generic(&mut self, expr: ExprId) -> ExprId {
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
        self.simplifier.simplify_with_stats(expr, generic_opts).0
    }

    fn fold_numeric_islands(&mut self, expr: ExprId) -> ExprId {
        super::numeric_islands::fold_numeric_islands(&mut self.simplifier.context, expr)
    }

    fn is_numeric_zero(&mut self, expr: ExprId) -> bool {
        is_numeric_zero(&self.simplifier.context, expr)
    }

    fn contains_variable(&mut self, expr: ExprId) -> bool {
        contains_variable(&self.simplifier.context, expr)
    }

    fn render_expr(&mut self, expr: ExprId) -> String {
        solver_render_expr(&self.simplifier.context, expr)
    }
}

struct EngineVerifyEquivalenceRuntime<'a> {
    simplifier: &'a mut Simplifier,
}

impl VerifySolutionRuntime for EngineVerifyEquivalenceRuntime<'_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn simplify_expr(&mut self, expr: ExprId) -> ExprId {
        self.simplifier.simplify(expr).0
    }

    fn are_equivalent(&mut self, lhs: ExprId, rhs: ExprId) -> bool {
        self.simplifier.are_equivalent(lhs, rhs)
    }
}

struct EngineVerifySetRuntime<'a> {
    simplifier: &'a mut Simplifier,
    equation: &'a Equation,
    var: &'a str,
}

impl VerifySolutionSetRuntime for EngineVerifySetRuntime<'_> {
    fn verify_discrete(&mut self, solution: ExprId) -> VerifyStatus {
        verify_solution(self.simplifier, self.equation, self.var, solution)
    }
}

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
    let mut runtime = EngineSolutionCheckRuntime { simplifier };
    verify_solution_with_core_runtime(&mut runtime, equation, var, solution)
}

/// Fast equivalence-based verifier used by solve strategy filtering.
///
/// This keeps legacy behavior for strategy-level post-filtering:
/// substitute both equation sides and check semantic equivalence after simplify.
pub(crate) fn verify_solution_by_equivalence(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solution: ExprId,
) -> bool {
    let mut runtime = EngineVerifyEquivalenceRuntime { simplifier };
    verify_solution_with_equivalence_core_runtime(&mut runtime, equation, var, solution)
}

/// Verify a solution set, handling all SolutionSet variants.
pub fn verify_solution_set(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solutions: &SolutionSet,
) -> VerifyResult {
    let mut runtime = EngineVerifySetRuntime {
        simplifier,
        equation,
        var,
    };
    verify_solution_set_with_runtime(solutions, &mut runtime)
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
