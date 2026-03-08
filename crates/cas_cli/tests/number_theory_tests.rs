use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_solver::rules::arithmetic::{CombineConstantsRule, MulOneRule};
use cas_solver::rules::exponents::ProductPowerRule;
use cas_solver::rules::number_theory::NumberTheoryRule;
use cas_solver::Simplifier;

fn create_nt_simplifier() -> Simplifier {
    let mut simplifier = Simplifier::new();
    simplifier.add_rule(Box::new(NumberTheoryRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier
}

#[test]
fn test_gcd() {
    let mut simplifier = create_nt_simplifier();
    // gcd(12, 18) -> 6
    let expr = parse("gcd(12, 18)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        ),
        "6"
    );
}

#[test]
fn test_lcm() {
    let mut simplifier = create_nt_simplifier();
    // lcm(4, 6) -> 12
    let expr = parse("lcm(4, 6)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        ),
        "12"
    );
}

#[test]
fn test_mod() {
    let mut simplifier = create_nt_simplifier();
    // mod(10, 3) -> 1
    let expr = parse("mod(10, 3)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        ),
        "1"
    );

    // mod(-10, 3) -> 2 (Euclidean)
    let expr2 = parse("mod(-10, 3)", &mut simplifier.context).unwrap();
    let (res2, _) = simplifier.simplify(expr2);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res2
            }
        ),
        "2"
    );
}

#[test]
fn test_prime_factors() {
    let mut simplifier = create_nt_simplifier();
    // factors(12) -> 2^2 * 3
    let expr = parse("factors(12)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let out = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    // Order might vary: 2^2 * 3 or 3 * 2^2
    assert!(out.contains("2^2"));
    assert!(out.contains("3"));
    assert!(out.contains("*"));
}

#[test]
fn test_prime_factors_large() {
    let mut simplifier = create_nt_simplifier();
    // factors(100) -> 2^2 * 5^2
    let expr = parse("factors(100)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let out = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert!(out.contains("2^2"));
    assert!(out.contains("5^2"));
}

#[test]
fn test_factorial() {
    let mut simplifier = create_nt_simplifier();
    // fact(5) -> 120
    let expr = parse("fact(5)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        ),
        "120"
    );

    // 5! -> 120
    let expr = parse("5!", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        ),
        "120"
    );

    // 0! -> 1
    let expr = parse("0!", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        ),
        "1"
    );
}

#[test]
fn test_infix_mod() {
    let mut simplifier = create_nt_simplifier();
    // 10 mod 3 -> 1
    let expr = parse("10 mod 3", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        ),
        "1"
    );

    // (20 + 5) mod 7 -> 25 mod 7 -> 4
    let expr = parse("(20 + 5) mod 7", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        ),
        "4"
    );
}

#[test]
fn test_combinatorics() {
    let mut simplifier = create_nt_simplifier();

    // choose(5, 2) -> 10
    let expr = parse("choose(5, 2)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        ),
        "10"
    );

    // choose(5, 0) -> 1
    let expr = parse("choose(5, 0)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        ),
        "1"
    );

    // choose(5, 5) -> 1
    let expr = parse("choose(5, 5)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        ),
        "1"
    );

    // perm(5, 2) -> 20
    let expr = parse("perm(5, 2)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        ),
        "20"
    );

    // perm(5, 5) -> 120 (5!)
    let expr = parse("perm(5, 5)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: res
            }
        ),
        "120"
    );
}
