use cas_ast::{Context, Expr};
use cas_formatter::DisplayExpr;
use cas_math::poly_gcd_structural::poly_gcd_structural;
use cas_math::poly_modp_calls::try_eval_poly_mul_modp_stats_call_with_limit_policy;
use cas_math::poly_modp_conv::{check_poly_equal_modp_expr, DEFAULT_PRIME};
use cas_parser::parse;
use num_bigint::BigInt;

#[test]
fn structural_poly_gcd_simple_common_factor() {
    let mut ctx = Context::new();

    let x = ctx.var("x");
    let one = ctx.num(1);
    let x_plus_1 = ctx.add(Expr::Add(x, one));

    let y = ctx.var("y");
    let two = ctx.num(2);
    let y_plus_2 = ctx.add(Expr::Add(y, two));
    let a = ctx.add(Expr::Mul(x_plus_1, y_plus_2));

    let z = ctx.var("z");
    let three = ctx.num(3);
    let z_plus_3 = ctx.add(Expr::Add(z, three));
    let b = ctx.add(Expr::Mul(x_plus_1, z_plus_3));

    let gcd = poly_gcd_structural(&mut ctx, a, b);
    assert_eq!(gcd, x_plus_1);
}

#[test]
fn structural_poly_gcd_with_powers() {
    let mut ctx = Context::new();

    let x = ctx.var("x");
    let one = ctx.num(1);
    let x_plus_1 = ctx.add(Expr::Add(x, one));

    let three = ctx.num(3);
    let pow3 = ctx.add(Expr::Pow(x_plus_1, three));

    let two = ctx.num(2);
    let pow2 = ctx.add(Expr::Pow(x_plus_1, two));

    let gcd = poly_gcd_structural(&mut ctx, pow3, pow2);
    let rendered = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: gcd
        }
    );
    assert!(rendered.contains("x"));
    assert!(rendered.contains("^2"));
}

#[test]
fn structural_poly_gcd_no_common_returns_one() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");

    let gcd = poly_gcd_structural(&mut ctx, x, y);
    match ctx.get(gcd) {
        Expr::Number(n) => {
            assert_eq!(*n, num_rational::BigRational::from_integer(BigInt::from(1)));
        }
        other => panic!("Expected numeric 1, got {:?}", other),
    }
}

#[test]
fn modp_equality_same_polynomial() {
    let mut ctx = Context::new();
    let a = parse("x + 1", &mut ctx).expect("parse a");
    let b = parse("1 + x", &mut ctx).expect("parse b");
    let result = check_poly_equal_modp_expr(&ctx, a, b, DEFAULT_PRIME).expect("modp equality");
    assert!(result);
}

#[test]
fn modp_equality_different_polynomial() {
    let mut ctx = Context::new();
    let a = parse("x + 1", &mut ctx).expect("parse a");
    let b = parse("x + 2", &mut ctx).expect("parse b");
    let result = check_poly_equal_modp_expr(&ctx, a, b, DEFAULT_PRIME).expect("modp equality");
    assert!(!result);
}

#[test]
fn poly_mul_modp_stats_rewrite_returns_stats_function() {
    let mut ctx = Context::new();
    let expr = parse("poly_mul_modp((x+1)^2, (x-1)^2)", &mut ctx).expect("parse call");

    let rewritten = try_eval_poly_mul_modp_stats_call_with_limit_policy(
        &mut ctx,
        expr,
        DEFAULT_PRIME,
        100_000,
        |_est, _lim| {},
    )
    .expect("poly_mul_modp rewrite");

    match ctx.get(rewritten.stats_expr) {
        Expr::Function(fn_id, args) => {
            assert_eq!(ctx.sym_name(*fn_id), "poly_mul_stats");
            assert_eq!(args.len(), 4);
        }
        other => panic!("Expected poly_mul_stats function, got {:?}", other),
    }
}
