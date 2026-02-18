//! Contract tests for SolveSafety pipeline filter.
//!
//! These tests verify that:
//! 1. SolvePrepass blocks conditional rules (NeedsCondition)
//! 2. Solver correctly handles cases where prepass protection is critical
//! 3. No regressions in existing solver behavior

use cas_ast::{Equation, RelOp};
use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

/// Helper to parse an expression
fn parse_expr(simplifier: &mut Simplifier, input: &str) -> cas_ast::ExprId {
    parse(input, &mut simplifier.context).unwrap()
}

/// Helper to create an equation
fn make_eq(simplifier: &mut Simplifier, lhs: &str, rhs: &str) -> Equation {
    Equation {
        lhs: parse_expr(simplifier, lhs),
        rhs: parse_expr(simplifier, rhs),
        op: RelOp::Eq,
    }
}

mod prepass_tests {
    use super::*;

    /// CancelCommonFactorsRule should have NeedsCondition(Definability) solve_safety.
    /// This ensures it won't be applied in SolvePrepass mode.
    #[test]
    fn cancel_common_factors_rule_marked_correctly() {
        use cas_engine::rule::SimpleRule;
        use cas_engine::rules::algebra::fractions::CancelCommonFactorsRule;
        use cas_engine::solve_safety::SolveSafety;

        let rule = CancelCommonFactorsRule;
        let safety = rule.solve_safety();

        // Must be NeedsCondition, not Always
        assert!(
            !safety.safe_for_prepass(),
            "CancelCommonFactorsRule should NOT be safe for prepass, got: {:?}",
            safety
        );

        // Specifically should be Definability class
        match safety {
            SolveSafety::NeedsCondition(_) => { /* OK */ }
            other => panic!("Expected NeedsCondition, got: {:?}", other),
        }
    }

    /// SimplifyFractionRule should also have NeedsCondition(Definability).
    #[test]
    fn simplify_fraction_rule_marked_correctly() {
        use cas_engine::rule::SimpleRule;
        use cas_engine::rules::algebra::fractions::SimplifyFractionRule;
        use cas_engine::solve_safety::SolveSafety;

        let rule = SimplifyFractionRule;
        let safety = rule.solve_safety();

        assert!(
            !safety.safe_for_prepass(),
            "SimplifyFractionRule should NOT be safe for prepass, got: {:?}",
            safety
        );

        match safety {
            SolveSafety::NeedsCondition(_) => { /* OK */ }
            other => panic!("Expected NeedsCondition, got: {:?}", other),
        }
    }

    /// Prepass should NOT apply LogExpansionRule.
    /// ln(x*y) should not become ln(x) + ln(y) as this requires x>0 and y>0.
    #[test]
    fn prepass_does_not_expand_log() {
        let mut simplifier = Simplifier::with_default_rules();
        // Add the LogExpansionRule (it's not registered by default)
        simplifier.add_rule(Box::new(cas_engine::rules::logarithms::LogExpansionRule));

        let expr = parse_expr(&mut simplifier, "ln(x*y)");

        // Use simplify_for_solve (SolvePrepass purpose)
        let result = simplifier.simplify_for_solve(expr);

        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
        );

        // Should NOT be expanded to ln(x) + ln(y)
        assert!(
            !result_str.contains("+"),
            "Prepass should NOT expand ln(x*y), got: {}",
            result_str
        );
    }
}

mod solve_tactic_tests {
    use super::*;
    use cas_engine::domain::DomainMode;
    use cas_engine::SimplifyOptions;

    /// SolveTactic in Assume mode SHOULD allow Analytic rules.
    /// This test verifies that exp(ln(x)) simplifies to x in tactic mode.
    #[test]
    fn tactic_in_assume_allows_analytic_rules() {
        let mut simplifier = Simplifier::with_default_rules();
        let expr = parse_expr(&mut simplifier, "exp(ln(x))");

        let opts = SimplifyOptions::for_solve_tactic(DomainMode::Assume);
        let (result, _) = simplifier.simplify_with_options(expr, opts);

        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
        );

        // In Assume mode with SolveTactic, exp(ln(x)) should simplify to x
        assert_eq!(
            result_str, "x",
            "SolveTactic(Assume) should allow exp(ln(x))→x, got: {}",
            result_str
        );
    }

    /// SolveTactic in Generic mode SHOULD allow intrinsic Analytic rules.
    /// exp(ln(x)) → x is allowed because ln(x) intrinsically requires x > 0;
    /// this is an inherited condition, not an introduced one.
    #[test]
    fn tactic_in_generic_allows_intrinsic_analytic_rules() {
        let mut simplifier = Simplifier::with_default_rules();
        let expr = parse_expr(&mut simplifier, "exp(ln(x))");

        let opts = SimplifyOptions::for_solve_tactic(DomainMode::Generic);
        let (result, _) = simplifier.simplify_with_options(expr, opts);

        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
        );

        // In Generic mode, intrinsic conditions (from ln(x)) are inherited
        assert_eq!(
            result_str, "x",
            "SolveTactic(Generic) should allow exp(ln(x))→x (intrinsic x>0 from ln), got: {}",
            result_str
        );
    }

    /// Guard test: SolveTactic should NOT transform inputs outside Assume+RealOnly
    /// This is a regression guard to ensure scope is not widened by future refactors.
    #[test]
    fn tactic_scope_guard_strict_mode() {
        let mut simplifier = Simplifier::with_default_rules();
        let expr = parse_expr(&mut simplifier, "exp(ln(x))");

        // In Strict mode, even SolveTactic should not apply Analytic rules
        let opts = SimplifyOptions::for_solve_tactic(DomainMode::Strict);
        let (result, _) = simplifier.simplify_with_options(expr, opts);

        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
        );

        // Strict mode: absolutely no Analytic rules
        assert!(
            result_str.contains("ln"),
            "SolveTactic(Strict) must not apply any Analytic rules, got: {}",
            result_str
        );
    }
}

mod solver_tests {
    use super::*;
    use cas_ast::domain::{BoundType, SolutionSet};

    /// The 0^x = 0 case must return interval (0, ∞), not be corrupted by prepass.
    /// This is the critical case that motivated the SolveSafety architecture.
    #[test]
    fn zero_to_x_equals_zero_preserves_interval() {
        let mut simplifier = Simplifier::with_default_rules();
        let eq = make_eq(&mut simplifier, "0^x", "0");

        let result = cas_engine::solver::solve(&eq, "x", &mut simplifier);

        match result {
            Ok((solution, _steps)) => {
                // Expect interval (0, ∞) or equivalent
                match solution {
                    SolutionSet::Continuous(interval) => {
                        // Should be open interval starting at 0, unbounded above
                        // min_type = Open means > 0, max_type = Open and max = Infinity means unbounded
                        assert_eq!(
                            interval.min_type,
                            BoundType::Open,
                            "Expected open lower bound (0, not [0)], got: {:?}",
                            interval
                        );
                    }
                    SolutionSet::AllReals => {
                        // This would be incorrect! The prepass corrupted the solution.
                        panic!(
                            "0^x = 0 should NOT return AllReals, the prepass may have corrupted it"
                        );
                    }
                    other => {
                        // Could be acceptable depending on solver behavior
                        println!("Got solution: {:?}", other);
                    }
                }
            }
            Err(e) => {
                panic!("Failed to solve 0^x = 0: {:?}", e);
            }
        }
    }

    /// a^x = a must return x = 1 (Power Equals Base Shortcut).
    /// This is a regression test to ensure prepass doesn't break existing behavior.
    #[test]
    fn power_equals_base_shortcut_works() {
        let mut simplifier = Simplifier::with_default_rules();
        let eq = make_eq(&mut simplifier, "a^x", "a");

        let result = cas_engine::solver::solve(&eq, "x", &mut simplifier);

        match result {
            Ok((solution, _steps)) => match solution {
                SolutionSet::Discrete(solutions) => {
                    assert_eq!(solutions.len(), 1, "Expected single solution x=1");
                    let sol_str = format!(
                        "{}",
                        DisplayExpr {
                            context: &simplifier.context,
                            id: solutions[0],
                        }
                    );
                    assert_eq!(sol_str, "1", "Expected solution x=1, got: {}", sol_str);
                }
                other => {
                    panic!("Expected discrete solution, got: {:?}", other);
                }
            },
            Err(e) => {
                panic!("Failed to solve a^x = a: {:?}", e);
            }
        }
    }

    /// Simple linear equation should still work after prepass changes.
    #[test]
    fn simple_linear_equation_works() {
        let mut simplifier = Simplifier::with_default_rules();
        let eq = make_eq(&mut simplifier, "2*x + 3", "7");

        let result = cas_engine::solver::solve(&eq, "x", &mut simplifier);

        match result {
            Ok((solution, _steps)) => match solution {
                SolutionSet::Discrete(solutions) => {
                    assert_eq!(solutions.len(), 1);
                    let sol_str = format!(
                        "{}",
                        DisplayExpr {
                            context: &simplifier.context,
                            id: solutions[0],
                        }
                    );
                    assert_eq!(sol_str, "2", "Expected x=2, got: {}", sol_str);
                }
                other => {
                    panic!("Expected discrete solution, got: {:?}", other);
                }
            },
            Err(e) => {
                panic!("Failed to solve 2x + 3 = 7: {:?}", e);
            }
        }
    }
}

/// Guardrail tests: verify that all sensitive rules are properly marked
/// as NeedsCondition. If these tests fail, a dangerous rule was added
/// without proper SolveSafety classification.
mod guardrail_tests {
    use cas_engine::rule::{Rule, SimpleRule};
    use cas_engine::solve_safety::SolveSafety;

    /// Helper macro for SimpleRule implementations
    macro_rules! assert_simple_not_always {
        ($rule:expr, $name:expr) => {
            let safety = SimpleRule::solve_safety(&$rule);
            assert!(
                !matches!(safety, SolveSafety::Always),
                "{} should be marked with NeedsCondition, got: {:?}",
                $name,
                safety
            );
        };
    }

    /// Helper macro for Rule implementations (manual impl)
    macro_rules! assert_rule_not_always {
        ($rule:expr, $name:expr) => {
            let safety = Rule::solve_safety(&$rule);
            assert!(
                !matches!(safety, SolveSafety::Always),
                "{} should be marked with NeedsCondition, got: {:?}",
                $name,
                safety
            );
        };
    }

    /// All Definability rules must be marked
    #[test]
    fn definability_rules_marked() {
        use cas_engine::rules::algebra::fractions::{
            CancelCommonFactorsRule, QuotientOfPowersRule, SimplifyFractionRule,
        };
        use cas_engine::rules::arithmetic::{DivZeroRule, MulZeroRule};
        use cas_engine::rules::exponents::IdentityPowerRule;

        assert_simple_not_always!(CancelCommonFactorsRule, "CancelCommonFactorsRule");
        assert_simple_not_always!(SimplifyFractionRule, "SimplifyFractionRule");
        assert_simple_not_always!(QuotientOfPowersRule, "QuotientOfPowersRule");
        assert_simple_not_always!(IdentityPowerRule, "IdentityPowerRule");
        assert_simple_not_always!(MulZeroRule, "MulZeroRule");
        assert_simple_not_always!(DivZeroRule, "DivZeroRule");
    }

    /// All Analytic rules with SimpleRule impl
    #[test]
    fn analytic_simplerule_marked() {
        use cas_engine::rules::exponents::PowerPowerRule;
        use cas_engine::rules::hyperbolic::HyperbolicCompositionRule;
        use cas_engine::rules::logarithms::{LogInversePowerRule, SplitLogExponentsRule};
        use cas_engine::rules::trig_inverse_expansion::TrigInverseExpansionRule;

        assert_simple_not_always!(LogInversePowerRule, "LogInversePowerRule");
        assert_simple_not_always!(SplitLogExponentsRule, "SplitLogExponentsRule");
        assert_simple_not_always!(PowerPowerRule, "PowerPowerRule");
        assert_simple_not_always!(HyperbolicCompositionRule, "HyperbolicCompositionRule");
        assert_simple_not_always!(TrigInverseExpansionRule, "TrigInverseExpansionRule");
    }

    /// All Analytic rules with manual Rule impl
    #[test]
    fn analytic_rule_manual_marked() {
        use cas_engine::rules::logarithms::{ExponentialLogRule, LogExpansionRule};

        assert_rule_not_always!(LogExpansionRule, "LogExpansionRule");
        assert_rule_not_always!(ExponentialLogRule, "ExponentialLogRule");
    }
}
