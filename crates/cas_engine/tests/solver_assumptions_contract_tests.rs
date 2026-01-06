//! Contract tests for PR-SCOPE-3.2: Solver Domain Inference in Output.
//!
//! These tests verify that the solver correctly derives implicit domain
//! conditions during solving (e.g., "positive(y)" for 2^x = y).
//!
//! NOTE: The original tests expected "assumptions" to be collected via
//! start/finish_assumption_collection(). However, with the V2.2+ domain
//! inference system, these conditions are now captured as "required conditions"
//! via derive_requires_from_equation(). This is the correct semantic:
//! - Required: structural domain facts derived from equation
//! - Assumptions: runtime assertions made during simplification (different system)

use cas_ast::{Equation, Expr, RelOp};
use cas_engine::domain::DomainMode;
use cas_engine::implicit_domain::ImplicitCondition;
use cas_engine::semantics::{AssumeScope, ValueDomain};
use cas_engine::solver::{solve_with_options, take_solver_required, SolverOptions};
use cas_engine::Engine;

fn make_opts(mode: DomainMode, scope: AssumeScope) -> SolverOptions {
    SolverOptions {
        value_domain: ValueDomain::RealOnly,
        domain_mode: mode,
        assume_scope: scope,
        budget: cas_engine::solver::SolveBudget::default(),
    }
}

fn setup_engine() -> Engine {
    Engine::new()
}

// =============================================================================
// Test 1: Assume mode derives positive(rhs) as required condition
// =============================================================================

#[test]
fn assume_mode_derives_positive_rhs_required() {
    // 2^x = y in any mode should derive required condition: positive(y)
    // This is because 2^x > 0 for all real x, so the equation implies y > 0
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let two = ctx.num(2);
    let x = ctx.var("x");
    let y = ctx.var("y");
    let pow = ctx.add(Expr::Pow(two, x));

    let eq = Equation {
        lhs: pow,
        rhs: y,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Assume, AssumeScope::Real);

    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    // Get required conditions from solver (saved in TLS by clear_current_domain_env)
    let required = take_solver_required();

    // Verify solve succeeded
    assert!(result.is_ok(), "Solve should succeed in Assume mode");

    // Verify required condition was derived
    let has_positive_y = required.iter().any(|cond| {
        if let ImplicitCondition::Positive(expr_id) = cond {
            let expr_str = cas_ast::DisplayExpr {
                context: &engine.simplifier.context,
                id: *expr_id,
            }
            .to_string();
            expr_str == "y"
        } else {
            false
        }
    });

    assert!(
        has_positive_y,
        "Should have positive(y) in required conditions, got: {:?}",
        required
    );
}

// =============================================================================
// Test 2: Strict mode has no additional requirements for positive base+rhs
// =============================================================================

#[test]
fn strict_mode_no_extra_requirements_for_literals() {
    // 2^x = 5 in Strict mode - no extra requirements (both provably positive)
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let two = ctx.num(2);
    let x = ctx.var("x");
    let five = ctx.num(5);
    let pow = ctx.add(Expr::Pow(two, x));

    let eq = Equation {
        lhs: pow,
        rhs: five,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Strict, AssumeScope::Real);

    let _result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    // Get required conditions
    let required = take_solver_required();

    // Strict mode with literal positive numbers: no extra Positive requirements
    // (the numbers are already proven positive, no need to require)
    let has_positive_5 = required.iter().any(|cond| {
        if let ImplicitCondition::Positive(expr_id) = cond {
            let expr_str = cas_ast::DisplayExpr {
                context: &engine.simplifier.context,
                id: *expr_id,
            }
            .to_string();
            expr_str == "5"
        } else {
            false
        }
    });

    assert!(
        !has_positive_5,
        "Should NOT require positive(5) for literal, got: {:?}",
        required
    );
}

// =============================================================================
// Test 3: Required conditions are deduplicated
// =============================================================================

#[test]
fn required_conditions_are_deduplicated() {
    // Solving an equation should not produce duplicate required conditions
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    // 2^x = y
    let two = ctx.num(2);
    let x = ctx.var("x");
    let y = ctx.var("y");
    let pow = ctx.add(Expr::Pow(two, x));

    let eq = Equation {
        lhs: pow,
        rhs: y,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Assume, AssumeScope::Real);

    let _ = solve_with_options(&eq, "x", &mut engine.simplifier, opts);
    let required = take_solver_required();

    // Count unique positive(y) conditions
    let positive_y_count = required
        .iter()
        .filter(|cond| {
            if let ImplicitCondition::Positive(expr_id) = cond {
                let expr_str = cas_ast::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: *expr_id,
                }
                .to_string();
                expr_str == "y"
            } else {
                false
            }
        })
        .count();

    assert!(
        positive_y_count <= 1,
        "Should have at most 1 positive(y) condition (deduped), got {}",
        positive_y_count
    );
}

// =============================================================================
// Test 4: Nested solves - each gets own required conditions
// =============================================================================

#[test]
fn nested_solves_have_isolated_requirements() {
    // Each solve call should have its own isolated required conditions
    // This tests the TLS mechanism for storing/clearing requirements

    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    // Outer equation: 2^x = y (requires positive(y))
    let two = ctx.num(2);
    let x = ctx.var("x");
    let y = ctx.var("y");
    let pow_outer = ctx.add(Expr::Pow(two, x));

    let eq_outer = Equation {
        lhs: pow_outer,
        rhs: y,
        op: RelOp::Eq,
    };

    // Inner equation: 3^z = w (requires positive(w))
    let three = ctx.num(3);
    let z = ctx.var("z");
    let w = ctx.var("w");
    let pow_inner = ctx.add(Expr::Pow(three, z));

    let eq_inner = Equation {
        lhs: pow_inner,
        rhs: w,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Assume, AssumeScope::Real);

    // Solve outer
    let _ = solve_with_options(&eq_outer, "x", &mut engine.simplifier, opts);
    let outer_required = take_solver_required();

    // Solve inner
    let _ = solve_with_options(&eq_inner, "z", &mut engine.simplifier, opts);
    let inner_required = take_solver_required();

    // Outer should have positive(y)
    let outer_has_y = outer_required.iter().any(|cond| {
        if let ImplicitCondition::Positive(id) = cond {
            cas_ast::DisplayExpr {
                context: &engine.simplifier.context,
                id: *id,
            }
            .to_string()
                == "y"
        } else {
            false
        }
    });

    // Inner should have positive(w)
    let inner_has_w = inner_required.iter().any(|cond| {
        if let ImplicitCondition::Positive(id) = cond {
            cas_ast::DisplayExpr {
                context: &engine.simplifier.context,
                id: *id,
            }
            .to_string()
                == "w"
        } else {
            false
        }
    });

    assert!(
        outer_has_y,
        "Outer solve should have positive(y), got: {:?}",
        outer_required
    );
    assert!(
        inner_has_w,
        "Inner solve should have positive(w), got: {:?}",
        inner_required
    );

    // Inner should NOT have positive(y) (from outer solve)
    let inner_has_y = inner_required.iter().any(|cond| {
        if let ImplicitCondition::Positive(id) = cond {
            cas_ast::DisplayExpr {
                context: &engine.simplifier.context,
                id: *id,
            }
            .to_string()
                == "y"
        } else {
            false
        }
    });

    assert!(
        !inner_has_y,
        "Inner solve should NOT have positive(y) from outer, got: {:?}",
        inner_required
    );
}
