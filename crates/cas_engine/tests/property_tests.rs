use cas_engine::Simplifier;
use cas_ast::Expr;
use proptest::prelude::*;

mod strategies;

proptest! {
    #[test]
    fn test_round_trip_parse(expr in strategies::arb_expr()) {
        let s = expr.to_string();
        // Parse it back
        let parsed = cas_parser::parse(&s);
        // It should parse successfully
        prop_assert!(parsed.is_ok(), "Failed to parse: {}", s);
    }

    #[test]
    fn test_identity_add_zero(expr in strategies::arb_expr()) {
        let simplifier = Simplifier::with_default_rules();
        // expr + 0 should simplify to expr (or equivalent)
        let input = Expr::add(expr.clone(), Expr::num(0));
        
        // simplify(expr + 0) == simplify(expr)
        let s1 = simplifier.simplify(input);
        let s2 = simplifier.simplify(expr);
        prop_assert_eq!(s1.0, s2.0);
    }

    #[test]
    fn test_identity_mul_one(expr in strategies::arb_expr()) {
        let simplifier = Simplifier::with_default_rules();
        // expr * 1 should simplify to expr
        let input = Expr::mul(expr.clone(), Expr::num(1));
        let s1 = simplifier.simplify(input);
        let s2 = simplifier.simplify(expr);
        prop_assert_eq!(s1.0, s2.0);
    }
    
    #[test]
    fn test_idempotency(expr in strategies::arb_expr()) {
        let simplifier = Simplifier::with_default_rules();
        let s1 = simplifier.simplify(expr);
        let s2 = simplifier.simplify(s1.0.clone());
        prop_assert_eq!(s1.0, s2.0);
    }
}
