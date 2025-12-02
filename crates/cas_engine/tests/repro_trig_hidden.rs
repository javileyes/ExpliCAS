#[cfg(test)]
mod tests {
    use cas_engine::Simplifier;
    use cas_parser::parse;
    use cas_ast::DisplayExpr;

    #[test]
    fn test_repro_trig_identity_hidden() {
        let mut simplifier = Simplifier::new();
        cas_engine::rules::algebra::register(&mut simplifier);
        cas_engine::rules::polynomial::register(&mut simplifier);
        cas_engine::rules::arithmetic::register(&mut simplifier);
        cas_engine::rules::exponents::register(&mut simplifier);
        cas_engine::rules::trigonometry::register(&mut simplifier);
        cas_engine::rules::canonicalization::register(&mut simplifier);

        // sin(x)^4 - cos(x)^4 - (sin(x)^2 - cos(x)^2) -> 0
        let expr_str = "sin(x)^4 - cos(x)^4 - (sin(x)^2 - cos(x)^2)";
        let expr = parse(expr_str, &mut simplifier.context).unwrap();
        let (res, _) = simplifier.simplify(expr);
        
        let res_str = format!("{}", DisplayExpr { context: &simplifier.context, id: res });
        println!("Result: {}", res_str);
        assert_eq!(res_str, "0");
    }
}
