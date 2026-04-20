#[cfg(test)]
mod engine_tests {
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;
    use cas_solver::runtime::Simplifier;

    fn create_standard_simplifier() -> Simplifier {
        let mut simplifier = Simplifier::new();
        cas_solver::runtime::rules::canonicalization::register(&mut simplifier);
        cas_solver::runtime::rules::arithmetic::register(&mut simplifier);
        cas_solver::runtime::rules::algebra::register(&mut simplifier);
        cas_solver::runtime::rules::trigonometry::register(&mut simplifier);
        cas_solver::runtime::rules::logarithms::register(&mut simplifier);
        cas_solver::runtime::rules::exponents::register(&mut simplifier);
        cas_solver::runtime::rules::functions::register(&mut simplifier);
        cas_solver::runtime::rules::polynomial::register(&mut simplifier);

        // Ensure DistributeRule is present (it might be in polynomial::register, but let's be sure)
        // Actually, polynomial::register usually adds DistributeRule.
        // Let's check if we need to add it explicitly or if it's already there.
        // If it's already there, we don't need to add it.
        // But the test was adding it manually, implying it might not be in the default register set?
        // Or maybe it was just to be safe.
        // Let's assume standard registration is enough, but if not, we add it here ONCE.

        simplifier
    }

    #[test]
    fn test_ramanujan_shadow_engine() {
        let mut simplifier = create_standard_simplifier();

        let expr = parse(
            "sqrt(3 + 2*sqrt(2)) - (1 + sqrt(2))",
            &mut simplifier.context,
        )
        .unwrap();
        println!(
            "Parsed Ramanujan expr: {}",
            DisplayExpr {
                context: &simplifier.context,
                id: expr
            }
        );
        let (simplified, _) = simplifier.simplify(expr);
        let res = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: simplified
            }
        );
        println!("Ramanujan Result: {}", res);
        assert_eq!(res, "0");
    }

    #[test]
    fn test_logarithmic_mirror_engine() {
        let mut simplifier = create_standard_simplifier();

        // x^(1/ln(x)) - exp(1)
        let expr = parse("x^(1/ln(x)) - exp(1)", &mut simplifier.context).unwrap();
        println!("Parsed Log Mirror expr: {:?}", simplifier.context.get(expr));
        let (simplified, _) = simplifier.simplify(expr);
        let res = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: simplified
            }
        );
        println!("Log Mirror Result: {}", res);
        assert_eq!(res, "0");
    }

    #[test]
    fn test_triple_angle_engine() {
        let mut simplifier = create_standard_simplifier();

        // sin(3*x) - (3*sin(x) - 4*sin(x)^3)
        let expr = parse(
            "sin(3*x) - (3*sin(x) - 4*sin(x)^3)",
            &mut simplifier.context,
        )
        .unwrap();
        let (simplified, _) = simplifier.simplify(expr);
        let res = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: simplified
            }
        );
        println!("Triple Angle Result: {}", res);

        // With CanonicalizeNegationRule enabled, this now fully simplifies to 0!
        // sin(3x) = 3sin(x) - 4sin³(x) is a trig identity, so sin(3x) - (3sin(x) - 4sin^3(x)) = 0
        assert_eq!(res, "0");
    }

    #[test]
    fn test_triple_sine_quotient_engine() {
        let mut simplifier = create_standard_simplifier();

        let expr = parse("sin(3*x)/sin(x) - 2*cos(2*x) - 1", &mut simplifier.context).unwrap();
        let (simplified, _) = simplifier.simplify(expr);
        let res = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: simplified
            }
        );

        assert_eq!(res, "0");
    }

    #[test]
    fn test_full_mixed_identity_engine() {
        let mut simplifier = create_standard_simplifier();

        let expr = parse(
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
            &mut simplifier.context,
        )
        .unwrap();
        let (simplified, _) = simplifier.simplify(expr);
        let res = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: simplified
            }
        );

        assert_eq!(res, "0");
    }

    #[test]
    fn test_exact_additive_pair_chain_arithmetic_runtime() {
        let mut simplifier = Simplifier::new();
        cas_solver::runtime::rules::canonicalization::register(&mut simplifier);
        cas_solver::runtime::rules::arithmetic::register(&mut simplifier);

        let expr = parse("2*cos(2*x)+1-2*cos(2*x)", &mut simplifier.context).unwrap();
        let (simplified, _) = simplifier.simplify(expr);
        let res = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: simplified
            }
        );

        assert_eq!(res, "1");
    }

    #[test]
    fn test_exact_additive_pair_chain_single_rule_runtime() {
        let mut simplifier = Simplifier::new();
        simplifier.set_collect_steps(false);
        simplifier.add_rule(Box::new(
            cas_solver::runtime::rules::arithmetic::CancelExactAdditivePairsRule,
        ));

        let expr = parse("2*cos(2*x)+1-2*cos(2*x)", &mut simplifier.context).unwrap();
        let (simplified, _) = simplifier.simplify(expr);
        let res = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: simplified
            }
        );

        assert_eq!(res, "1");
    }
}
