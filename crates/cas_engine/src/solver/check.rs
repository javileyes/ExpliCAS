//! Solution verification module.
//!
//! Verifies solver solutions by substituting back into the original equation
//! and checking if the result simplifies to zero (using Strict mode).
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

use cas_ast::{DisplayExpr, Equation, Expr, ExprId, SolutionSet};
use num_traits::Zero;

use crate::engine::Simplifier;

/// Result of verifying a single solution.
#[derive(Debug, Clone)]
pub enum VerifyStatus {
    /// Solution verified: equation simplifies to 0 after substitution
    Verified,
    /// Solution could not be verified (residual remains)
    Unverifiable {
        /// The residual expression that didn't simplify to 0
        residual: ExprId,
        /// Human-readable reason
        reason: String,
    },
    /// Solution type not checkable (intervals, AllReals, residual)
    NotCheckable {
        /// Reason why verification is not possible
        reason: &'static str,
    },
}

/// Summary of verification for a solution set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifySummary {
    /// All solutions verified
    AllVerified,
    /// Some solutions verified, some not
    PartiallyVerified,
    /// No solutions verified
    NoneVerified,
    /// Solution type not checkable
    NotCheckable,
    /// Empty solution set (trivially verified)
    Empty,
}

/// Result of verifying an entire solution set.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// Status for each discrete solution (if applicable)
    pub solutions: Vec<(ExprId, VerifyStatus)>,
    /// Overall summary
    pub summary: VerifySummary,
    /// Guard under which verification was performed (for Conditional)
    pub guard_description: Option<String>,
}

/// Verify a single solution by substituting into the equation.
///
/// # Algorithm
/// 1. Substitute `var := solution` in both `lhs` and `rhs`
/// 2. Compute `diff = lhs_sub - rhs_sub`
/// 3. Simplify with **Strict mode** (no assumptions)
/// 4. If result is `0` → `Verified`, otherwise → `Unverifiable`
pub fn verify_solution(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solution: ExprId,
) -> VerifyStatus {
    // Step 1: Substitute solution into lhs and rhs
    let lhs_sub = substitute(simplifier, equation.lhs, var, solution);
    let rhs_sub = substitute(simplifier, equation.rhs, var, solution);

    // Step 2: Compute difference lhs - rhs
    let diff = simplifier.context.add(Expr::Sub(lhs_sub, rhs_sub));

    // Step 3: Simplify with Strict mode (no assumptions allowed)
    let opts = crate::SimplifyOptions {
        shared: crate::phase::SharedSemanticConfig {
            semantics: crate::semantics::EvalConfig {
                domain_mode: crate::domain::DomainMode::Strict,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let (simplified, _, _) = simplifier.simplify_with_stats(diff, opts);

    // Step 4: Check if result is 0
    match simplifier.context.get(simplified) {
        Expr::Number(n) if n.is_zero() => VerifyStatus::Verified,
        _ => {
            let residual_str = DisplayExpr {
                context: &simplifier.context,
                id: simplified,
            }
            .to_string();
            VerifyStatus::Unverifiable {
                residual: simplified,
                reason: format!("residual: {}", residual_str),
            }
        }
    }
}

/// Verify a solution set, handling all SolutionSet variants.
pub fn verify_solution_set(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solutions: &SolutionSet,
) -> VerifyResult {
    match solutions {
        SolutionSet::Empty => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::Empty,
            guard_description: None,
        },

        SolutionSet::Discrete(sols) => {
            let mut results = Vec::with_capacity(sols.len());
            let mut verified_count = 0;

            for &sol in sols {
                let status = verify_solution(simplifier, equation, var, sol);
                if matches!(status, VerifyStatus::Verified) {
                    verified_count += 1;
                }
                results.push((sol, status));
            }

            let summary = if results.is_empty() {
                VerifySummary::Empty
            } else if verified_count == results.len() {
                VerifySummary::AllVerified
            } else if verified_count > 0 {
                VerifySummary::PartiallyVerified
            } else {
                VerifySummary::NoneVerified
            };

            VerifyResult {
                solutions: results,
                summary,
                guard_description: None,
            }
        }

        SolutionSet::AllReals => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some("not checkable (infinite set: all reals)".to_string()),
        },

        SolutionSet::Continuous(_interval) => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some("not checkable (continuous interval)".to_string()),
        },

        SolutionSet::Union(_intervals) => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some("not checkable (union of intervals)".to_string()),
        },

        SolutionSet::Residual(_expr) => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some("unverifiable (residual expression)".to_string()),
        },

        SolutionSet::Conditional(cases) => {
            // For conditional, verify each case separately
            let mut all_results = Vec::new();
            let mut has_verified = false;
            let mut has_not_checkable = false;

            for case in cases {
                // Get the solution set from this case's SolveResult
                let case_solutions = &case.then.solutions;

                // Recursively verify this case's solutions
                let case_result = verify_solution_set(simplifier, equation, var, case_solutions);

                match case_result.summary {
                    VerifySummary::AllVerified | VerifySummary::PartiallyVerified => {
                        has_verified = true;
                    }
                    VerifySummary::NotCheckable => {
                        has_not_checkable = true;
                    }
                    _ => {}
                }

                all_results.extend(case_result.solutions);
            }

            let summary = if has_verified && !has_not_checkable {
                VerifySummary::AllVerified
            } else if has_verified {
                VerifySummary::PartiallyVerified
            } else if has_not_checkable {
                VerifySummary::NotCheckable
            } else {
                VerifySummary::NoneVerified
            };

            VerifyResult {
                solutions: all_results,
                summary,
                guard_description: None,
            }
        }
    }
}

/// Substitute a variable with a value in an expression.
fn substitute(simplifier: &mut Simplifier, expr: ExprId, var: &str, value: ExprId) -> ExprId {
    let expr_data = simplifier.context.get(expr).clone();

    match expr_data {
        Expr::Variable(sym_id) if simplifier.context.sym_name(sym_id) == var => value,
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => expr,

        Expr::Add(a, b) => {
            let a_sub = substitute(simplifier, a, var, value);
            let b_sub = substitute(simplifier, b, var, value);
            simplifier.context.add(Expr::Add(a_sub, b_sub))
        }
        Expr::Sub(a, b) => {
            let a_sub = substitute(simplifier, a, var, value);
            let b_sub = substitute(simplifier, b, var, value);
            simplifier.context.add(Expr::Sub(a_sub, b_sub))
        }
        Expr::Mul(a, b) => {
            let a_sub = substitute(simplifier, a, var, value);
            let b_sub = substitute(simplifier, b, var, value);
            simplifier.context.add(Expr::Mul(a_sub, b_sub))
        }
        Expr::Div(a, b) => {
            let a_sub = substitute(simplifier, a, var, value);
            let b_sub = substitute(simplifier, b, var, value);
            simplifier.context.add(Expr::Div(a_sub, b_sub))
        }
        Expr::Pow(a, b) => {
            let a_sub = substitute(simplifier, a, var, value);
            let b_sub = substitute(simplifier, b, var, value);
            simplifier.context.add(Expr::Pow(a_sub, b_sub))
        }
        Expr::Neg(a) => {
            let a_sub = substitute(simplifier, a, var, value);
            simplifier.context.add(Expr::Neg(a_sub))
        }
        Expr::Function(name, args) => {
            let args_sub: Vec<_> = args
                .iter()
                .map(|&arg| substitute(simplifier, arg, var, value))
                .collect();
            simplifier.context.add(Expr::Function(name, args_sub))
        }
        Expr::Matrix { rows, cols, data } => {
            let data_sub: Vec<_> = data
                .iter()
                .map(|&elem| substitute(simplifier, elem, var, value))
                .collect();
            simplifier.context.add(Expr::Matrix {
                rows,
                cols,
                data: data_sub,
            })
        }
        Expr::Hold(inner) => {
            let inner_sub = substitute(simplifier, inner, var, value);
            simplifier.context.add(Expr::Hold(inner_sub))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
