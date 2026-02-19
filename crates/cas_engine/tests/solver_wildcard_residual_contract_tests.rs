//! Contract tests for PR-SCOPE-3.1: Wildcard Residual for NeedsComplex.
//!
//! These tests verify that in Assume+Wildcard mode, exponential equations
//! with negative bases return a structured residual instead of an error.

use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_engine::solver::{solve_with_display_steps, SolverOptions};
use cas_engine::DomainMode;
use cas_engine::Engine;
use cas_engine::{AssumeScope, ValueDomain};

fn make_opts(mode: DomainMode, scope: AssumeScope) -> SolverOptions {
    SolverOptions {
        value_domain: ValueDomain::RealOnly,
        domain_mode: mode,
        assume_scope: scope,
        budget: cas_engine::solver::SolveBudget::default(),
        ..Default::default()
    }
}

fn setup_engine() -> Engine {
    Engine::new()
}

// =============================================================================
// T1: Wildcard mode returns Residual + warning (not error)
// =============================================================================

#[test]
fn wildcard_negative_base_returns_residual() {
    // (-2)^x = 5 in Assume + Wildcard mode -> Residual (not error)
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let neg_two = ctx.num(-2);
    let x = ctx.var("x");
    let five = ctx.num(5);
    let pow = ctx.add(Expr::Pow(neg_two, x));

    let eq = Equation {
        lhs: pow,
        rhs: five,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Assume, AssumeScope::Wildcard);
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);

    match result {
        Ok((SolutionSet::Residual(residual_expr), steps, _diagnostics)) => {
            // Verify it's a residual
            let expr_str = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: residual_expr
                }
            );
            // The residual should contain "solve"
            assert!(
                expr_str.contains("solve") || expr_str.contains("__eq__"),
                "Residual should contain 'solve' function, got: {}",
                expr_str
            );

            // Check that steps contain the warning message
            if !steps.is_empty() {
                let has_warning = steps
                    .iter()
                    .any(|s| s.description.contains("complex") || s.description.contains("preset"));
                assert!(has_warning, "Steps should contain warning about complex");
            }
        }
        Ok((other, _, _)) => {
            panic!(
                "Wildcard mode should return Residual for negative base, got: {:?}",
                other
            );
        }
        Err(e) => {
            panic!(
                "Wildcard mode should NOT error for negative base, got error: {}",
                e
            );
        }
    }
}

// =============================================================================
// T2: No garbage ln(-2) in result
// =============================================================================

#[test]
fn wildcard_residual_no_ln_negative() {
    // (-2)^x = 5 residual should NOT contain ln(-2)
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let neg_two = ctx.num(-2);
    let x = ctx.var("x");
    let five = ctx.num(5);
    let pow = ctx.add(Expr::Pow(neg_two, x));

    let eq = Equation {
        lhs: pow,
        rhs: five,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Assume, AssumeScope::Wildcard);
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);

    if let Ok((SolutionSet::Residual(residual_expr), _, _)) = result {
        let expr_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &engine.simplifier.context,
                id: residual_expr
            }
        );
        // NO-GARBAGE INVARIANT: Must not contain ln(-...)
        assert!(
            !expr_str.contains("ln(-"),
            "Residual should NOT contain ln(-), got: {}",
            expr_str
        );
        assert!(
            !expr_str.contains("undefined"),
            "Residual should NOT contain 'undefined', got: {}",
            expr_str
        );
    }
    // If not Residual, the test passes (T1 already checks this)
}

// =============================================================================
// T3: AssumeScope::Real still errors (not residual)
// =============================================================================

#[test]
fn assume_real_negative_base_does_not_return_residual() {
    // (-2)^x = 5 in Assume + Real mode -> should NOT return Residual
    // (should error or skip to let other strategy handle)
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let neg_two = ctx.num(-2);
    let x = ctx.var("x");
    let five = ctx.num(5);
    let pow = ctx.add(Expr::Pow(neg_two, x));

    let eq = Equation {
        lhs: pow,
        rhs: five,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Assume, AssumeScope::Real);
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);

    match result {
        Ok((SolutionSet::Residual(_), _, _)) => {
            panic!("AssumeScope::Real should NOT return Residual for negative base");
        }
        _ => {
            // Any other result (error or other solution set) is acceptable
        }
    }
}

// =============================================================================
// T4: Strict/Generic mode still errors (not residual)
// =============================================================================

#[test]
fn strict_mode_negative_base_does_not_return_residual() {
    // (-2)^x = 5 in Strict mode -> should NOT return Residual
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let neg_two = ctx.num(-2);
    let x = ctx.var("x");
    let five = ctx.num(5);
    let pow = ctx.add(Expr::Pow(neg_two, x));

    let eq = Equation {
        lhs: pow,
        rhs: five,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Strict, AssumeScope::Real);
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);

    match result {
        Ok((SolutionSet::Residual(_), _, _)) => {
            panic!("Strict mode should NOT return Residual for negative base");
        }
        _ => {
            // Any other result (error or other solution set) is acceptable
        }
    }
}
