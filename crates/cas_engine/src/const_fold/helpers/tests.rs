use super::*;

#[test]
fn test_is_constant_literal() {
    let mut ctx = Context::new();

    let num = ctx.num(42);
    assert!(is_constant_literal(&ctx, num));

    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
    assert!(is_constant_literal(&ctx, pi));

    let x = ctx.var("x");
    assert!(!is_constant_literal(&ctx, x));
}

#[test]
fn test_fold_sqrt_positive_perfect() {
    let mut ctx = Context::new();
    let four = ctx.num(4);

    let result = fold_sqrt(&mut ctx, four, ValueDomain::RealOnly);
    assert!(result.is_some());

    // Should be 2
    if let Some(r) = result {
        assert!(matches!(ctx.get(r), Expr::Number(n) if n == &BigRational::from_integer(2.into())));
    }
}

#[test]
fn test_fold_sqrt_negative_real() {
    let mut ctx = Context::new();
    let neg_one = ctx.num(-1);

    let result = fold_sqrt(&mut ctx, neg_one, ValueDomain::RealOnly);
    assert!(result.is_some());

    // Should be undefined
    if let Some(r) = result {
        assert!(matches!(
            ctx.get(r),
            Expr::Constant(cas_ast::Constant::Undefined)
        ));
    }
}

#[test]
fn test_fold_sqrt_negative_complex() {
    let mut ctx = Context::new();
    let neg_one = ctx.num(-1);

    let result = fold_sqrt(&mut ctx, neg_one, ValueDomain::ComplexEnabled);
    assert!(result.is_some());

    // Should be i * sqrt(1) = Mul(i, sqrt(1))
    if let Some(r) = result {
        assert!(matches!(ctx.get(r), Expr::Mul(_, _)));
    }
}

#[test]
fn test_fold_mul_i_i() {
    let mut ctx = Context::new();
    let i1 = ctx.add(Expr::Constant(cas_ast::Constant::I));
    let i2 = ctx.add(Expr::Constant(cas_ast::Constant::I));

    let result = fold_mul_imaginary(&mut ctx, i1, i2, ValueDomain::ComplexEnabled);
    assert!(result.is_some());

    // Should be -1
    if let Some(r) = result {
        assert!(
            matches!(ctx.get(r), Expr::Number(n) if n == &BigRational::from_integer((-1).into()))
        );
    }
}
