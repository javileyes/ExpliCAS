#[cfg(test)]
mod tests {
    use cas_ast::DisplayExpr;
    use cas_engine::Simplifier;
    use cas_parser::parse;

    #[test]
    fn test_repro_trig_identity_hidden() {
        let mut simplifier = Simplifier::new();
        cas_engine::rules::algebra::register(&mut simplifier);
        cas_engine::rules::polynomial::register(&mut simplifier);
        cas_engine::rules::arithmetic::register(&mut simplifier);
        cas_engine::rules::exponents::register(&mut simplifier);
        cas_engine::rules::trigonometry::register(&mut simplifier);
        cas_engine::rules::canonicalization::register(&mut simplifier);

        // sin(x)^4 - cos(x)^4 - (sin(x)^2 - cos(x)^2)
        // This SHOULD simplify to 0 via: sin^4 - cos^4 = (sin^2-cos^2) (using sin²+cos²=1)
        // so the expression becomes (sin^2-cos^2) - (sin^2-cos^2) = 0
        //
        // Now correctly simplified by TrigEvenPowerDifferenceRule which reduces
        // sin^4 - cos^4 → sin^2 - cos^2 (degree-reducing, loop-safe)
        let expr_str = "sin(x)^4 - cos(x)^4 - (sin(x)^2 - cos(x)^2)";
        let expr = parse(expr_str, &mut simplifier.context).unwrap();
        let (res, _) = simplifier.simplify(expr);

        let res_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        );
        println!("Result: {}", res_str);

        // Fixed: TrigEvenPowerDifferenceRule now correctly simplifies this to 0
        assert_eq!(res_str, "0", "Expression should simplify to 0");
    }
}
