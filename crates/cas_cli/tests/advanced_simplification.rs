#[cfg(test)]
mod engine_tests {
    use cas_engine::Simplifier;
    use cas_parser::parse;
    use cas_ast::{DisplayExpr, Expr};

    #[test]
    fn test_ramanujan_shadow_engine() {
        let mut simplifier = Simplifier::new();
        cas_engine::rules::canonicalization::register(&mut simplifier);
        cas_engine::rules::arithmetic::register(&mut simplifier);
        cas_engine::rules::algebra::register(&mut simplifier);
        cas_engine::rules::trigonometry::register(&mut simplifier);
        cas_engine::rules::logarithms::register(&mut simplifier);
        cas_engine::rules::exponents::register(&mut simplifier);
        cas_engine::rules::functions::register(&mut simplifier);
        cas_engine::rules::polynomial::register(&mut simplifier);
        
        let expr = parse("sqrt(3 + 2*sqrt(2)) - (1 + sqrt(2))", &mut simplifier.context).unwrap();
        println!("Parsed Ramanujan expr: {}", DisplayExpr { context: &simplifier.context, id: expr });
        let (simplified, _) = simplifier.simplify(expr);
        let res = format!("{}", DisplayExpr { context: &simplifier.context, id: simplified });
        println!("Ramanujan Result: {}", res);
        assert_eq!(res, "0");
    }

    #[test]
    fn test_logarithmic_mirror_engine() {
        let mut simplifier = Simplifier::new();
        cas_engine::rules::canonicalization::register(&mut simplifier);
        cas_engine::rules::arithmetic::register(&mut simplifier);
        cas_engine::rules::algebra::register(&mut simplifier);
        cas_engine::rules::trigonometry::register(&mut simplifier);
        cas_engine::rules::logarithms::register(&mut simplifier);
        cas_engine::rules::exponents::register(&mut simplifier);
        cas_engine::rules::polynomial::register(&mut simplifier);
        
        // x^(1/ln(x)) - exp(1)
        let expr = parse("x^(1/ln(x)) - exp(1)", &mut simplifier.context).unwrap();
        println!("Parsed Log Mirror expr: {:?}", simplifier.context.get(expr));
        let (simplified, _) = simplifier.simplify(expr);
        let res = format!("{}", DisplayExpr { context: &simplifier.context, id: simplified });
        println!("Log Mirror Result: {}", res);
        assert_eq!(res, "0");
    }

    #[test]
    fn test_triple_angle_engine() {
        let mut simplifier = Simplifier::new();
        cas_engine::rules::canonicalization::register(&mut simplifier);
        cas_engine::rules::arithmetic::register(&mut simplifier);
        cas_engine::rules::algebra::register(&mut simplifier);
        cas_engine::rules::trigonometry::register(&mut simplifier);
        cas_engine::rules::exponents::register(&mut simplifier);
        cas_engine::rules::polynomial::register(&mut simplifier); // For distribution
        
        // Enable distribution for expansion
        simplifier.add_rule(Box::new(cas_engine::rules::polynomial::DistributeRule));

        // sin(3*x) - (3*sin(x) - 4*sin(x)^3)
        let expr = parse("sin(3*x) - (3*sin(x) - 4*sin(x)^3)", &mut simplifier.context).unwrap();
        let (simplified, _) = simplifier.simplify(expr);
        let res = format!("{}", DisplayExpr { context: &simplifier.context, id: simplified });
        println!("Triple Angle Result: {}", res);
        assert_eq!(res, "0");
    }
}
