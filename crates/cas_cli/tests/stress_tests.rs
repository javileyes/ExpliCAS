use cas_ast::{Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_solver::rules::algebra::SimplifyFractionRule;
use cas_solver::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule};
use cas_solver::rules::canonicalization::{
    CanonicalizeAddRule, CanonicalizeMulRule, CanonicalizeNegationRule, CanonicalizeRootRule,
};
use cas_solver::rules::exponents::{
    EvaluatePowerRule, IdentityPowerRule, PowerPowerRule, ProductPowerRule,
};
use cas_solver::rules::functions::EvaluateAbsRule;
use cas_solver::rules::logarithms::{EvaluateLogRule, ExponentialLogRule};
use cas_solver::rules::polynomial::{AnnihilationRule, CombineLikeTermsRule, DistributeRule};
use cas_solver::rules::trigonometry::{EvaluateTrigRule, PythagoreanIdentityRule};
use cas_solver::solve;
use cas_solver::Simplifier;
use num_traits::Zero;

fn create_full_simplifier() -> Simplifier {
    let mut simplifier = Simplifier::new();
    // Canonicalization first
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeRootRule));

    // Evaluation
    simplifier.add_rule(Box::new(EvaluateAbsRule));
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(EvaluateLogRule));
    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(EvaluatePowerRule));

    // Exponents
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(IdentityPowerRule));

    // Polynomials
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(AnnihilationRule));

    // Algebra
    simplifier.add_rule(Box::new(SimplifyFractionRule));

    // Arithmetic Cleanup
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));

    simplifier
}

#[test]
fn test_deeply_nested_simplification() {
    // ((x+1)^2 + (x-1)^2) / (x^2 + 1)
    // Numerator: (x^2 + 2x + 1) + (x^2 - 2x + 1) = 2x^2 + 2 = 2(x^2 + 1)
    // Result: 2

    let mut simplifier = create_full_simplifier();
    // Note: We need DistributeRule to expand (x+1)^2, but currently DistributeRule might only handle a*(b+c).
    // Power expansion (a+b)^2 is not yet implemented as a rule!
    // This test is expected to fail or return unsimplified result if we don't have "ExpandPowerRule".
    // Let's see what happens.

    let input_str = "((x + 1)^2 + (x - 1)^2) / (x^2 + 1)";
    let input = parse(input_str, &mut simplifier.context).unwrap();
    let expected = parse("2", &mut simplifier.context).unwrap();

    // If it fails to simplify fully, it might remain as input.
    // If we want it to pass, we might need to implement ExpandPowerRule.
    // For stress testing, let's assert the ideal result and see.
    assert_equivalent(&mut simplifier, input, expected);
}

#[test]
fn test_mixed_transcendental() {
    // sin(ln(exp(x))) -> sin(x)
    let mut simplifier = create_full_simplifier();
    let input_str = "sin(ln(exp(x)))";
    let input = parse(input_str, &mut simplifier.context).unwrap();
    let expected_str = "sin(x)";
    let expected = parse(expected_str, &mut simplifier.context).unwrap();
    assert_equivalent(&mut simplifier, input, expected);
}

#[test]
fn test_rational_simplification() {
    // (x^2 - 1) / (x - 1) -> x + 1
    let mut simplifier = create_full_simplifier();
    let input_str = "(x^2 - 1) / (x - 1)";
    let input = parse(input_str, &mut simplifier.context).unwrap();
    let expected_str = "1 + x";
    let expected = parse(expected_str, &mut simplifier.context).unwrap();
    assert_equivalent(&mut simplifier, input, expected);
}

fn assert_equivalent(s: &mut Simplifier, expr1: ExprId, expr2: ExprId) {
    let (sim1, _) = s.simplify(expr1);
    let (sim2, _) = s.simplify(expr2);

    if s.are_equivalent(sim1, sim2) {
        return;
    }

    let diff = s.context.add(Expr::Sub(sim1, sim2));
    let (sim_diff, _) = s.simplify(diff);

    if let Expr::Number(n) = s.context.get(sim_diff) {
        if n.is_zero() {
            return;
        }
    }

    panic!(
        "Expressions not equivalent.\nExpr1: {}\nSim1: {}\nExpr2: {}\nSim2: {}\nDiff: {}",
        DisplayExpr {
            context: &s.context,
            id: expr1
        },
        DisplayExpr {
            context: &s.context,
            id: sim1
        },
        DisplayExpr {
            context: &s.context,
            id: expr2
        },
        DisplayExpr {
            context: &s.context,
            id: sim2
        },
        DisplayExpr {
            context: &s.context,
            id: sim_diff
        }
    );
}

#[test]
fn test_quadratic_solver() {
    // solve(x^2 - 4 = 0, x) -> x = 2 (or x = -2, solver might pick one or return set)
    // Our solver currently isolates. x^2 = 4 -> x = 4^(1/2) -> x = 2.
    // It usually returns the principal root.

    let mut simplifier = create_full_simplifier();
    let lhs = parse("x^2 - 4", &mut simplifier.context).unwrap();
    let rhs = parse("0", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    // Pre-simplify: x^2 - 4 = 0.
    // Solver needs to move 4.
    // Our solver handles "Sub(l, r) = RHS". If l has var (x^2), it adds r to RHS.
    // So x^2 = 4.
    // Then Pow(b, e) = RHS. b = RHS^(1/e). x = 4^(1/2).
    // Then simplify 4^(1/2) -> 2.

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(solutions) = result {
        assert!(!solutions.is_empty());
        assert!(!solutions.is_empty());
        let found = solutions.iter().any(|res_rhs| {
            let (final_rhs, _) = simplifier.simplify(*res_rhs);
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: final_rhs
                }
            ) == "2"
        });
        assert!(found, "Expected solution '2' not found");
    } else {
        panic!("Expected Discrete solution");
    }
}

#[test]
fn test_exponential_solver() {
    // solve(exp(2*x) - 1 = 0, x) -> x = 0
    let mut simplifier = create_full_simplifier();
    let lhs = parse("exp(2 * x) - 1", &mut simplifier.context).unwrap();
    let rhs = parse("0", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(solutions) = result {
        assert!(!solutions.is_empty());
        assert!(!solutions.is_empty());
        let found = solutions.iter().any(|res_rhs| {
            let (final_rhs, _) = simplifier.simplify(*res_rhs);
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: final_rhs
                }
            ) == "0"
        });
        assert!(found, "Expected solution '0' not found");
    } else {
        panic!("Expected Discrete solution");
    }
}
