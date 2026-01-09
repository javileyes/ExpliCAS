//! REPL Output Snapshot Tests (Issue #2)
//!
//! These tests capture the expected output of iconic equations to detect UX regressions.
//! Uses the `insta` crate for snapshot testing.
//!
//! ## Updating Snapshots
//!
//! When output changes intentionally:
//! ```bash
//! cargo insta test -p cas_engine --test repl_snapshots --accept
//! ```
//!
//! Or review interactively:
//! ```bash
//! cargo insta review
//! ```

use cas_ast::{DisplayExpr, SolutionSet};
use cas_engine::domain::DomainMode;
use cas_engine::semantics::{AssumeScope, ValueDomain};
use cas_engine::solver::{solve_with_display_steps, SolveBudget, SolverOptions};
use cas_engine::Engine;

/// Helper to solve an equation and format the result as a snapshot-friendly string
fn solve_and_format(eq_str: &str, var: &str, budget: usize, mode: DomainMode) -> String {
    let mut engine = Engine::new();
    let ctx = &mut engine.simplifier.context;

    // Parse equation
    let stmt = cas_parser::parse_statement(eq_str, ctx).expect("Failed to parse equation");
    let eq = match stmt {
        cas_parser::Statement::Equation(eq) => eq,
        _ => panic!("Expected equation, got expression"),
    };

    // Solve with options
    let opts = SolverOptions {
        value_domain: ValueDomain::RealOnly,
        domain_mode: mode,
        assume_scope: AssumeScope::Real,
        budget: SolveBudget {
            max_branches: budget,
            max_depth: 2,
        },
        ..Default::default()
    };

    let result = solve_with_display_steps(&eq, var, &mut engine.simplifier, opts);

    match result {
        Ok((solution_set, _steps)) => {
            format_solution_set(&engine.simplifier.context, &solution_set)
        }
        Err(e) => format!("ERROR: {}", e),
    }
}

/// Format solution set in a deterministic way for snapshots
fn format_solution_set(ctx: &cas_ast::Context, set: &SolutionSet) -> String {
    match set {
        SolutionSet::Empty => "Empty Set".to_string(),
        SolutionSet::AllReals => "All Real Numbers".to_string(),
        SolutionSet::Discrete(exprs) => {
            let s: Vec<String> = exprs
                .iter()
                .map(|e| {
                    DisplayExpr {
                        context: ctx,
                        id: *e,
                    }
                    .to_string()
                })
                .collect();
            format!("{{ {} }}", s.join(", "))
        }
        SolutionSet::Continuous(interval) => {
            let min_bracket = match interval.min_type {
                cas_ast::BoundType::Open => "(",
                cas_ast::BoundType::Closed => "[",
            };
            let max_bracket = match interval.max_type {
                cas_ast::BoundType::Open => ")",
                cas_ast::BoundType::Closed => "]",
            };
            format!(
                "{}{}, {}{}",
                min_bracket,
                DisplayExpr {
                    context: ctx,
                    id: interval.min
                },
                DisplayExpr {
                    context: ctx,
                    id: interval.max
                },
                max_bracket
            )
        }
        SolutionSet::Union(intervals) => {
            let parts: Vec<String> = intervals
                .iter()
                .map(|i| {
                    let min_bracket = match i.min_type {
                        cas_ast::BoundType::Open => "(",
                        cas_ast::BoundType::Closed => "[",
                    };
                    let max_bracket = match i.max_type {
                        cas_ast::BoundType::Open => ")",
                        cas_ast::BoundType::Closed => "]",
                    };
                    format!(
                        "{}{}, {}{}",
                        min_bracket,
                        DisplayExpr {
                            context: ctx,
                            id: i.min
                        },
                        DisplayExpr {
                            context: ctx,
                            id: i.max
                        },
                        max_bracket
                    )
                })
                .collect();
            parts.join(" U ")
        }
        SolutionSet::Residual(expr) => {
            format!(
                "Residual: {}",
                DisplayExpr {
                    context: ctx,
                    id: *expr
                }
            )
        }
        SolutionSet::Conditional(cases) => {
            let mut lines = vec!["Conditional:".to_string()];
            for case in cases {
                let cond_str = case.when.display_with_context(ctx);
                let sol_str = format_solution_set(ctx, &case.then.solutions);
                if case.when.is_otherwise() {
                    lines.push(format!("  otherwise: {}", sol_str));
                } else {
                    lines.push(format!("  if {}: {}", cond_str, sol_str));
                }
            }
            lines.join("\n")
        }
    }
}

// =============================================================================
// Iconic Equation Snapshots
// =============================================================================

/// Test 1: a^x = a with budget=2 (Conditional 3 cases)
#[test]
fn snapshot_solve_a_pow_x_eq_a_conditional() {
    let result = solve_and_format("a^x = a", "x", 2, DomainMode::Strict);
    insta::assert_snapshot!(result);
}

/// Test 2: a^x = a with budget=1 (fallback to {1})
#[test]
fn snapshot_solve_a_pow_x_eq_a_fallback() {
    let result = solve_and_format("a^x = a", "x", 1, DomainMode::Generic);
    insta::assert_snapshot!(result);
}

/// Test 3: 0^x = 0 (interval x > 0)
#[test]
fn snapshot_solve_zero_pow_x_eq_zero() {
    let result = solve_and_format("0^x = 0", "x", 1, DomainMode::Generic);
    insta::assert_snapshot!(result);
}

/// Test 4: 2^x = 8 (x = 3)
#[test]
fn snapshot_solve_two_pow_x_eq_8() {
    let result = solve_and_format("2^x = 8", "x", 1, DomainMode::Generic);
    insta::assert_snapshot!(result);
}

/// Test 5: x^2 - 4 = 0 (quadratic)
#[test]
fn snapshot_solve_quadratic_simple() {
    let result = solve_and_format("x^2 - 4 = 0", "x", 1, DomainMode::Generic);
    insta::assert_snapshot!(result);
}

/// Test 6: x + 2 = 5 (simple linear)
#[test]
fn snapshot_solve_linear_simple() {
    let result = solve_and_format("x + 2 = 5", "x", 1, DomainMode::Generic);
    insta::assert_snapshot!(result);
}

/// Test 7: 2*x = 10 (simple linear multiplication)
#[test]
fn snapshot_solve_linear_mult() {
    let result = solve_and_format("2*x = 10", "x", 1, DomainMode::Generic);
    insta::assert_snapshot!(result);
}

/// Test 8: a^x = a^2 (equal bases pattern)
#[test]
fn snapshot_solve_equal_bases() {
    let result = solve_and_format("a^x = a^2", "x", 1, DomainMode::Strict);
    insta::assert_snapshot!(result);
}
