//! Solver Soundness Guard Tests
//!
//! Anti-unsoundness tests that verify the solver does not silently drop
//! analytic domain conditions. These tests enforce contracts like:
//!
//! - Log equations must carry positivity requirements
//! - `sqrt(x^2) = x` must not be unconditional AllReals
//! - Cancel pipeline must not erase domain constraints
//!
//! The contracts are intentionally robust: they check "at least one
//! relevant condition exists" rather than exact predicates, to avoid
//! coupling to internal representation (e.g., Positive(x) vs Positive(x*y)).

use cas_ast::SolutionSet;
use cas_engine::semantics::ValueDomain;
use cas_engine::solver::{solve_with_display_steps, SolverOptions};
use cas_engine::DomainMode;
use cas_engine::ImplicitCondition;
use cas_engine::Simplifier;
use cas_parser::{parse_statement, Statement};

/// Solve an equation string in Strict mode, returning (SolutionSet, required conditions).
fn solve_strict(eq_src: &str, var: &str) -> (SolutionSet, Vec<ImplicitCondition>, Simplifier) {
    let mut simplifier = Simplifier::with_default_rules();
    let stmt = parse_statement(eq_src, &mut simplifier.context).expect("parse failed");
    let eq = match stmt {
        Statement::Equation(eq) => eq,
        _ => panic!("expected equation, got expression"),
    };

    let opts = SolverOptions {
        value_domain: ValueDomain::RealOnly,
        domain_mode: DomainMode::Strict,
        budget: Default::default(),
        detailed_steps: false,
        ..Default::default()
    };

    let (sols, _steps, diagnostics) =
        solve_with_display_steps(&eq, var, &mut simplifier, opts).expect("solve failed");
    (sols, diagnostics.required, simplifier)
}

#[allow(dead_code)]
/// Check if any required condition involves positivity (Positive variant).
fn has_any_positive(req: &[ImplicitCondition]) -> bool {
    req.iter()
        .any(|c| matches!(c, ImplicitCondition::Positive(_)))
}

#[allow(dead_code)]
/// Check if any required condition involves non-negativity (NonNegative variant).
fn has_any_nonnegative(req: &[ImplicitCondition]) -> bool {
    req.iter()
        .any(|c| matches!(c, ImplicitCondition::NonNegative(_)))
}

#[allow(dead_code)]
/// Check if any required condition involves positivity OR non-negativity.
fn has_any_domain_restriction(req: &[ImplicitCondition]) -> bool {
    has_any_positive(req) || has_any_nonnegative(req)
}

// =============================================================================
// (A) Log product identity — must carry positivity
// =============================================================================
//
// ln(x*y) = ln(x) + ln(y) is only valid when x > 0 AND y > 0.
// The solver should NOT return unconditional AllReals with empty required.

#[test]
fn log_product_identity_requires_conditions() {
    let (sols, req, _s) = solve_strict("ln(x*y) = ln(x) + ln(y)", "x");

    // Contract: must NOT be "unconditional AllReals with empty required"
    if matches!(sols, SolutionSet::AllReals) && req.is_empty() {
        panic!(
            "UNSOUND: AllReals with no required conditions for ln(x*y) = ln(x) + ln(y). \
             Expected at least one Positive(...) condition."
        );
    }

    // If AllReals, must have positivity required
    if matches!(sols, SolutionSet::AllReals) {
        assert!(
            has_any_positive(&req),
            "AllReals must carry positivity conditions (got: {:?})",
            req
        );
    }
    // Other outcomes (Conditional, Residual, Discrete) are acceptable
    // as long as they carry domain info or restrict solutions appropriately.
}

// =============================================================================
// (B) Log tautology — AllReals is fine, but must have positivity
// =============================================================================
//
// ln(x*y) = ln(x*y) is a tautology, so AllReals is correct.
// But the equation only makes sense where x*y > 0.

#[test]
fn log_tautology_still_requires_domain() {
    let (sols, req, _s) = solve_strict("ln(x*y) = ln(x*y)", "x");

    // AllReals is the expected outcome for a tautology
    assert!(
        matches!(sols, SolutionSet::AllReals),
        "Expected AllReals for tautology ln(x*y) = ln(x*y), got {:?}",
        sols
    );

    // But it must carry positivity conditions (definability domain)
    assert!(
        has_any_positive(&req),
        "Tautology ln(x*y) = ln(x*y) must still carry Positive(...) requirement, \
         got empty required. The equation is only valid where x*y > 0."
    );
}

// =============================================================================
// (C) ln(sqrt(x)*y) = ln(x)/2 + ln(y) — must have domain conditions
// =============================================================================
//
// This involves both sqrt (x ≥ 0) and ln (argument > 0).

#[test]
fn ln_sqrt_product_requires_conditions() {
    let (sols, req, _s) = solve_strict("ln(sqrt(x)*y) = ln(x)/2 + ln(y)", "x");

    // Contract: must have at least one domain restriction
    if matches!(sols, SolutionSet::AllReals) && req.is_empty() {
        panic!(
            "UNSOUND: AllReals with no conditions for ln(sqrt(x)*y) = ln(x)/2 + ln(y). \
             Expected NonNegative or Positive conditions."
        );
    }
}

// =============================================================================
// (D) ln(x^2) = 2*ln(x) — requires x > 0 in reals
// =============================================================================
//
// LHS: ln(x^2) is defined for all x ≠ 0 (since x^2 > 0 always).
// RHS: 2*ln(x) requires x > 0.
// So the identity requires x > 0 (RHS is more restrictive).

#[test]
fn ln_pow_requires_base_positive_in_reals() {
    let (sols, req, _s) = solve_strict("ln(x^2) = 2*ln(x)", "x");

    // Contract: must have positivity or domain restriction
    if matches!(sols, SolutionSet::AllReals) && req.is_empty() {
        panic!(
            "UNSOUND: AllReals with no conditions for ln(x^2) = 2*ln(x). \
             RHS requires x > 0."
        );
    }
}

// =============================================================================
// (E) sqrt(x^2) = x — must NOT be unconditional AllReals
// =============================================================================
//
// sqrt(x^2) = |x| in reals, so sqrt(x^2) = x only holds for x ≥ 0.
// The solver should return Conditional, Interval [0,∞), or AllReals
// with NonNegative/Positive conditions — but NOT empty AllReals.

#[test]
fn sqrt_square_not_unconditional_identity() {
    let (sols, req, _s) = solve_strict("sqrt(x^2) = x", "x");

    // Contract: must not be unconditional AllReals with empty conditions
    if matches!(sols, SolutionSet::AllReals) && req.is_empty() {
        panic!(
            "UNSOUND: AllReals with no conditions for sqrt(x^2) = x. \
             This identity requires x >= 0. Expected NonNegative condition \
             or Conditional/Interval solution."
        );
    }
    // Acceptable outcomes:
    // - AllReals with NonNegative(x) or Positive(x)
    // - Continuous interval [0, ∞)
    // - Conditional solution
    // - Residual (solver gives up)
}

// =============================================================================
// (F) Cancel pipeline must not erase log domain
// =============================================================================
//
// ln(x*y) + 1 = ln(x) + ln(y) + 1
// The "+1" terms should cancel, leaving ln(x*y) = ln(x) + ln(y).
// After cancel, the domain conditions must survive.

#[test]
fn cancel_pipeline_must_not_drop_log_domain() {
    let (sols, req, _s) = solve_strict("ln(x*y) + 1 = ln(x) + ln(y) + 1", "x");

    // Same contract as test (A): the cancel of "+1" must not erase
    // the positivity requirements from the log terms.
    if matches!(sols, SolutionSet::AllReals) && req.is_empty() {
        panic!(
            "UNSOUND: Cancel pipeline dropped log domain conditions. \
             ln(x*y) + 1 = ln(x) + ln(y) + 1 must carry Positive conditions \
             after cancelling the '+1' terms."
        );
    }
}

// =============================================================================
// (G) abs(constant) = x must not produce spurious negative solutions
// =============================================================================
//
// abs(2) = x should simplify to 2 = x, giving {2}.
// It must NOT return {2, -2} (which would happen if isolate_abs
// splits into branches without the rhs ≥ 0 guard).

#[test]
fn abs_constant_eq_var_must_not_include_negative() {
    let (sols, _req, s) = solve_strict("abs(2) = x", "x");

    match &sols {
        SolutionSet::Discrete(vals) => {
            // Must contain 2
            let has_two = vals.iter().any(|v| {
                matches!(s.context.get(*v), cas_ast::Expr::Number(n)
                    if *n == num_rational::BigRational::from_integer(2.into()))
            });
            assert!(has_two, "Expected x = 2 in solutions, got {:?}", sols);

            // Must NOT contain -2
            let has_neg_two = vals.iter().any(|v| {
                matches!(s.context.get(*v), cas_ast::Expr::Number(n)
                    if *n == num_rational::BigRational::from_integer((-2).into()))
            });
            assert!(
                !has_neg_two,
                "UNSOUND: abs(2) = x returned -2 as a solution. \
                 abs(2) = 2, so x = 2 is the only valid solution."
            );
        }
        // Conditional with guard is also acceptable (sound but less ideal)
        SolutionSet::Conditional(_) => { /* sound */ }
        other => {
            panic!(
                "Unexpected solution type for abs(2) = x: {:?}. \
                 Expected Discrete({{2}}) or Conditional.",
                other
            );
        }
    }
}
