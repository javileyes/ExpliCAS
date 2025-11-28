#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_parser::parse;
    use cas_engine::Simplifier;

    #[test]
    fn test_canonicalize_add() {
        let mut simplifier = Simplifier::new();
        simplifier.add_rule(Box::new(CanonicalizeAddRule));

        // b + a -> a + b
        let expr = parse("b + a").unwrap();
        let (res, _) = simplifier.simplify(expr);
        assert_eq!(format!("{}", res), "a + b");

        // 1 + a -> 1 + a (Number < Variable)
        let expr = parse("1 + a").unwrap();
        let (res, _) = simplifier.simplify(expr);
        assert_eq!(format!("{}", res), "1 + a");

        // a + 1 -> 1 + a
        let expr = parse("a + 1").unwrap();
        let (res, _) = simplifier.simplify(expr);
        assert_eq!(format!("{}", res), "1 + a");
    }

    #[test]
    fn test_canonicalize_mul() {
        let mut simplifier = Simplifier::new();
        simplifier.add_rule(Box::new(CanonicalizeMulRule));

        // y * x -> x * y
        let expr = parse("y * x").unwrap();
        let (res, _) = simplifier.simplify(expr);
        assert_eq!(format!("{}", res), "x * y");

        // x * 2 -> 2 * x
        let expr = parse("x * 2").unwrap();
        let (res, _) = simplifier.simplify(expr);
        assert_eq!(format!("{}", res), "2 * x");
    }
}
