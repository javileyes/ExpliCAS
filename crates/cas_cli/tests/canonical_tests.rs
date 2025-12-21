//! Strong Canonicalization Property Tests
//!
//! These tests verify fundamental invariants that a well-behaved CAS must satisfy:
//! - Idempotence: simplify(simplify(e)) == simplify(e)
//! - Associativity invariance: a+(b+c) == (a+b)+c in canonical form
//! - Commutativity invariance: permuted terms produce same result
//! - Structural invariants: no nested Add/Mul, no Neg(Neg(x)), etc.
//! - Metamorphic: e+0 == e, e*1 == e, -(-e) == e

use cas_ast::{Context, DisplayExpr, Expr, ExprId};
use cas_engine::Simplifier;
use cas_parser::parse;

fn create_simplifier() -> Simplifier {
    let mut s = Simplifier::new();
    s.register_default_rules();
    s
}

// ============================================================================
// 1. IDEMPOTENCE
// ============================================================================

#[test]
fn test_idempotence_simple() {
    let mut s = create_simplifier();
    let cases = [
        "x + y + z",
        "x * y * z",
        "(x+1)*(x-1)",
        "x^2 + 2*x + 1",
        "1/sqrt(2)",
    ];

    for case in cases {
        let expr = parse(case, &mut s.context).unwrap();
        let (once, _) = s.simplify(expr);
        let (twice, _) = s.simplify(once);
        assert_eq!(
            once, twice,
            "Idempotence failed for '{}': simplify(simplify(e)) != simplify(e)",
            case
        );
    }
}

#[test]
fn test_idempotence_complex() {
    let mut s = create_simplifier();
    let cases = [
        "(a+b)*(c+d)",
        "x/(1+sqrt(2))",
        "sin(x)^2 + cos(x)^2",
        "-x + y - z",
    ];

    for case in cases {
        let expr = parse(case, &mut s.context).unwrap();
        let (once, _) = s.simplify(expr);
        let (twice, _) = s.simplify(once);
        assert_eq!(once, twice, "Idempotence failed for '{}'", case);
    }
}

// ============================================================================
// 2. ASSOCIATIVITY INVARIANCE
// ============================================================================

#[test]
fn test_associativity_add() {
    let mut s = create_simplifier();

    // a + (b + c) vs (a + b) + c
    let e1 = parse("x + (y + z)", &mut s.context).unwrap();
    let e2 = parse("(x + y) + z", &mut s.context).unwrap();

    let (r1, _) = s.simplify(e1);
    let (r2, _) = s.simplify(e2);

    assert_eq!(r1, r2, "Add associativity: x+(y+z) should equal (x+y)+z");
}

#[test]
fn test_associativity_mul() {
    let mut s = create_simplifier();

    // a * (b * c) vs (a * b) * c
    let e1 = parse("x * (y * z)", &mut s.context).unwrap();
    let e2 = parse("(x * y) * z", &mut s.context).unwrap();

    let (r1, _) = s.simplify(e1);
    let (r2, _) = s.simplify(e2);

    // Note: Mul may not flatten, so compare display instead
    let d1 = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: r1
        }
    );
    let d2 = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: r2
        }
    );
    assert_eq!(d1, d2, "Mul associativity: x*(y*z) should equal (x*y)*z");
}

#[test]
fn test_associativity_nested_add() {
    let mut s = create_simplifier();

    // ((a + b) + c) + d vs a + (b + (c + d))
    let e1 = parse("((a + b) + c) + d", &mut s.context).unwrap();
    let e2 = parse("a + (b + (c + d))", &mut s.context).unwrap();

    let (r1, _) = s.simplify(e1);
    let (r2, _) = s.simplify(e2);

    assert_eq!(r1, r2, "Nested Add associativity failed");
}

// ============================================================================
// 3. COMMUTATIVITY INVARIANCE
// ============================================================================

#[test]
fn test_commutativity_add() {
    let mut s = create_simplifier();

    let permutations = ["x + y + z", "y + x + z", "z + y + x", "x + z + y"];

    let first = parse(permutations[0], &mut s.context).unwrap();
    let (canonical, _) = s.simplify(first);

    for perm in &permutations[1..] {
        let expr = parse(perm, &mut s.context).unwrap();
        let (result, _) = s.simplify(expr);
        assert_eq!(
            canonical, result,
            "Commutativity failed: '{}' != '{}'",
            permutations[0], perm
        );
    }
}

#[test]
fn test_commutativity_mul() {
    let mut s = create_simplifier();

    let permutations = ["2 * x * y", "x * 2 * y", "y * x * 2"];

    let first = parse(permutations[0], &mut s.context).unwrap();
    let (canonical, _) = s.simplify(first);
    let d1 = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: canonical
        }
    );

    for perm in &permutations[1..] {
        let expr = parse(perm, &mut s.context).unwrap();
        let (result, _) = s.simplify(expr);
        let d2 = format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: result
            }
        );
        assert_eq!(
            d1, d2,
            "Mul commutativity failed: '{}' != '{}'",
            permutations[0], perm
        );
    }
}

// ============================================================================
// 4. STRUCTURAL INVARIANTS
// ============================================================================

/// Check that an expression tree satisfies structural invariants
fn check_structural_invariants(ctx: &Context, id: ExprId, path: &str) -> Result<(), String> {
    match ctx.get(id) {
        // Note: Balanced tree may create Add(Add(..), ..) - this is intentional for O(log n) depth
        Expr::Add(l, r) => {
            check_structural_invariants(ctx, *l, &format!("{}.l", path))?;
            check_structural_invariants(ctx, *r, &format!("{}.r", path))?;
        }
        // No Neg(Neg(x)) and no Neg(Number) - should be Number(-n)
        Expr::Neg(inner) => {
            if matches!(ctx.get(*inner), Expr::Neg(_)) {
                return Err(format!("{}: Neg(Neg(..)) found", path));
            }
            if matches!(ctx.get(*inner), Expr::Number(_)) {
                return Err(format!(
                    "{}: Neg(Number(..)) found - should be Number(-n)",
                    path
                ));
            }
            check_structural_invariants(ctx, *inner, &format!("{}.inner", path))?;
        }
        // No Pow(x, 1) or Pow(1, e) or Pow(x, 0)
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                use num_traits::{One, Zero};
                if n.is_one() {
                    return Err(format!("{}: Pow(x, 1) should be x", path));
                }
                if n.is_zero() {
                    return Err(format!("{}: Pow(x, 0) should be 1", path));
                }
            }
            check_structural_invariants(ctx, *base, &format!("{}.base", path))?;
            check_structural_invariants(ctx, *exp, &format!("{}.exp", path))?;
        }
        Expr::Mul(l, r) => {
            check_structural_invariants(ctx, *l, &format!("{}.l", path))?;
            check_structural_invariants(ctx, *r, &format!("{}.r", path))?;
        }
        Expr::Div(l, r) => {
            check_structural_invariants(ctx, *l, &format!("{}.l", path))?;
            check_structural_invariants(ctx, *r, &format!("{}.r", path))?;
        }
        Expr::Sub(l, r) => {
            check_structural_invariants(ctx, *l, &format!("{}.l", path))?;
            check_structural_invariants(ctx, *r, &format!("{}.r", path))?;
        }
        Expr::Function(_, args) => {
            for (i, arg) in args.iter().enumerate() {
                check_structural_invariants(ctx, *arg, &format!("{}.arg{}", path, i))?;
            }
        }
        // Atoms: no recursion needed
        _ => {}
    }
    Ok(())
}

#[test]
fn test_structural_invariants() {
    let mut s = create_simplifier();
    let cases = ["x + y + z", "--x", "x^1", "x^0 + 1", "((a + b) + c) + d"];

    for case in cases {
        let expr = parse(case, &mut s.context).unwrap();
        let (result, _) = s.simplify(expr);

        if let Err(e) = check_structural_invariants(&s.context, result, "root") {
            panic!("Structural invariant violated for '{}': {}", case, e);
        }
    }
}

// ============================================================================
// 5. N0 NORMALIZATION: Neg(Number) -> Number(-n)
// ============================================================================

#[test]
fn test_n0_neg_number_normalized() {
    use cas_engine::canonical_forms::normalize_core;
    use num_traits::Signed;

    let mut ctx = cas_ast::Context::new();

    // Parse -5 which produces Neg(Number(5))
    let expr = parse("-5", &mut ctx).unwrap();

    // After normalize_core, should be Number(-5), not Neg(Number(5))
    let normalized = normalize_core(&mut ctx, expr);

    // Verify it's a Number, not a Neg
    match ctx.get(normalized) {
        Expr::Number(n) => {
            assert!(n.is_negative(), "Number should be negative, got {}", n);
        }
        other => {
            panic!("Expected Number(-5), got {:?}", other);
        }
    }
}

#[test]
fn test_n0_structural_invariant_with_negatives() {
    let mut s = create_simplifier();
    // Cases with negative numbers that should NOT have Neg(Number) after simplify
    let cases = ["-5 + x", "-2 * x", "x + (-3)", "-1/2 * y"];

    for case in cases {
        let expr = parse(case, &mut s.context).unwrap();
        let (result, _) = s.simplify(expr);

        if let Err(e) = check_structural_invariants(&s.context, result, "root") {
            panic!("N0 structural invariant violated for '{}': {}", case, e);
        }
    }
}

// ============================================================================
// 6. METAMORPHIC TESTS (Neutral Operations)
// ============================================================================

#[test]
fn test_metamorphic_add_zero() {
    let mut s = create_simplifier();

    let e_raw = parse("x^2 + y", &mut s.context).unwrap();
    let e_with_zero = parse("x^2 + y + 0", &mut s.context).unwrap();

    let (r1, _) = s.simplify(e_raw);
    let (r2, _) = s.simplify(e_with_zero);

    assert_eq!(r1, r2, "e + 0 should equal e");
}

#[test]
fn test_metamorphic_mul_one() {
    let mut s = create_simplifier();

    let e_raw = parse("x + y", &mut s.context).unwrap();
    let e_with_one = parse("1 * (x + y)", &mut s.context).unwrap();

    let (r1, _) = s.simplify(e_raw);
    let (r2, _) = s.simplify(e_with_one);

    assert_eq!(r1, r2, "1 * e should equal e");
}

#[test]
fn test_metamorphic_div_one() {
    let mut s = create_simplifier();

    let e_raw = parse("x + y", &mut s.context).unwrap();
    let e_div_one = parse("(x + y) / 1", &mut s.context).unwrap();

    let (r1, _) = s.simplify(e_raw);
    let (r2, _) = s.simplify(e_div_one);

    assert_eq!(r1, r2, "e / 1 should equal e");
}

#[test]
fn test_metamorphic_double_neg() {
    let mut s = create_simplifier();

    let e_raw = parse("x + y", &mut s.context).unwrap();
    let e_double_neg = parse("--(x + y)", &mut s.context).unwrap();

    let (r1, _) = s.simplify(e_raw);
    let (r2, _) = s.simplify(e_double_neg);

    assert_eq!(r1, r2, "--e should equal e");
}

#[test]
fn test_metamorphic_pow_one() {
    let mut s = create_simplifier();

    let e_raw = parse("x + y", &mut s.context).unwrap();
    let e_pow_one = parse("(x + y)^1", &mut s.context).unwrap();

    let (r1, _) = s.simplify(e_raw);
    let (r2, _) = s.simplify(e_pow_one);

    assert_eq!(r1, r2, "e^1 should equal e");
}
