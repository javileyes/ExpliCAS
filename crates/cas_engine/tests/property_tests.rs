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

    #[test]
    fn test_constant_folding(expr in strategies::arb_expr()) {
        let simplifier = Simplifier::with_default_rules();
        let (simplified, _) = simplifier.simplify(expr);
        
        // Check that no Number op Number exists in the simplified expression
        // We can use a visitor or a recursive check.
        // Let's use a recursive check helper.
        fn check_no_constant_ops(expr: &Expr) -> bool {
            match expr {
                Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
                    if let (Expr::Number(_), Expr::Number(_)) = (l.as_ref(), r.as_ref()) {
                        return false;
                    }
                    check_no_constant_ops(l) && check_no_constant_ops(r)
                },
                Expr::Neg(e) => {
                    if let Expr::Number(_) = e.as_ref() {
                        // Neg(Number) is allowed if it's negative number representation, 
                        // but usually parser/engine puts negative in Number itself.
                        // Our engine might produce Neg(Number(5)) -> Number(-5).
                        // So Neg(Number) shouldn't exist ideally.
                        return false;
                    }
                    check_no_constant_ops(e)
                },
                Expr::Function(_, args) => args.iter().all(|a| check_no_constant_ops(a)),
                _ => true,
            }
        }
        
        prop_assert!(check_no_constant_ops(&simplified), "Constant folding failed: {}", simplified);
    }

    #[test]
    fn test_identity_preservation(expr in strategies::arb_expr()) {
        let simplifier = Simplifier::with_default_rules();
        
        // x * 0 -> 0
        let zero = Expr::num(0);
        let mul_zero = Expr::mul(expr.clone(), zero.clone());
        let (s_mul_zero, _) = simplifier.simplify(mul_zero);
        prop_assert_eq!(s_mul_zero, zero.clone());

        // x ^ 0 -> 1
        // Exception: 0^0 is undefined or 1 depending on convention. Our rule says 1.
        // But 0^0 might be tricky.
        let one = Expr::num(1);
        let pow_zero = Expr::pow(expr.clone(), zero.clone());
        let (s_pow_zero, _) = simplifier.simplify(pow_zero);
        prop_assert_eq!(s_pow_zero, one.clone());
        
        // x ^ 1 -> x
        let pow_one = Expr::pow(expr.clone(), one.clone());
        let (s_pow_one, _) = simplifier.simplify(pow_one);
        let (s_expr, _) = simplifier.simplify(expr.clone());
        prop_assert_eq!(s_pow_one, s_expr);
    }

    #[test]
    fn test_associativity_flattening(expr in strategies::arb_expr()) {
        let simplifier = Simplifier::with_default_rules();
        let (simplified, _) = simplifier.simplify(expr);
        
        // Check that we don't have (a + b) + c or (a * b) * c
        // The AssociativityRule should flatten these to a + (b + c) or similar canonical form.
        // Our rule is: (a + b) + c -> a + (b + c) (Right associative)
        // So we shouldn't see Add(Add(..), ..) as the LHS of an Add.
        
        fn check_right_associative(expr: &Expr) -> bool {
            match expr {
                Expr::Add(lhs, rhs) => {
                    if let Expr::Add(_, _) = lhs.as_ref() {
                        return false; // Found (a+b)+c
                    }
                    check_right_associative(lhs) && check_right_associative(rhs)
                },
                Expr::Mul(lhs, rhs) => {
                    if let Expr::Mul(_, _) = lhs.as_ref() {
                        return false; // Found (a*b)*c
                    }
                    check_right_associative(lhs) && check_right_associative(rhs)
                },
                Expr::Sub(lhs, rhs) | Expr::Div(lhs, rhs) | Expr::Pow(lhs, rhs) => {
                    check_right_associative(lhs) && check_right_associative(rhs)
                },
                Expr::Neg(e) => check_right_associative(e),
                Expr::Function(_, args) => args.iter().all(|a| check_right_associative(a)),
                _ => true,
            }
        }

        prop_assert!(check_right_associative(&simplified), "Associativity flattening failed: {}", simplified);
    }
}
