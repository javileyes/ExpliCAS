use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::Ratio;

/// Try to expand compositions like `sin(arctan(x))`.
///
/// Returns `(expanded_expr, description)` when a known identity matches.
pub fn expand_trig_inverse_composition(
    ctx: &mut Context,
    outer_name: &str,
    inner_name: &str,
    x: ExprId,
) -> Option<(ExprId, &'static str)> {
    for (outer_expected, inner_variants, transform, description) in EXPANSIONS {
        if outer_name == *outer_expected && inner_variants.contains(&inner_name) {
            let result = apply_transform(ctx, x, *transform);
            return Some((result, description));
        }
    }
    None
}

/// Build sqrt(expr) = expr^(1/2)
fn build_sqrt(ctx: &mut Context, expr: ExprId) -> ExprId {
    let half = ctx.add(Expr::Number(Ratio::new(BigInt::from(1), BigInt::from(2))));
    ctx.add(Expr::Pow(expr, half))
}

/// Build 1 - x²
fn build_one_minus_x_sq(ctx: &mut Context, x: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let x_sq = ctx.add(Expr::Pow(x, two));
    ctx.add(Expr::Sub(one, x_sq))
}

/// Build 1 + x²
fn build_one_plus_x_sq(ctx: &mut Context, x: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let x_sq = ctx.add(Expr::Pow(x, two));
    ctx.add(Expr::Add(one, x_sq))
}

/// Build x² - 1
fn build_x_sq_minus_one(ctx: &mut Context, x: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let x_sq = ctx.add(Expr::Pow(x, two));
    ctx.add(Expr::Sub(x_sq, one))
}

/// How to transform x for the result.
#[derive(Clone, Copy)]
enum Transform {
    /// x / √(base)
    XOverSqrt(Base),
    /// √(base) / x
    SqrtOverX(Base),
    /// 1 / √(base)
    OneOverSqrt(Base),
    /// 1 / x
    OneOverX,
    /// √(base)
    JustSqrt(Base),
}

/// Which polynomial to use inside sqrt.
#[derive(Clone, Copy)]
enum Base {
    OneMinus, // 1 - x²
    OnePlus,  // 1 + x²
    XMinus,   // x² - 1
}

fn build_base(ctx: &mut Context, x: ExprId, base: Base) -> ExprId {
    match base {
        Base::OneMinus => build_one_minus_x_sq(ctx, x),
        Base::OnePlus => build_one_plus_x_sq(ctx, x),
        Base::XMinus => build_x_sq_minus_one(ctx, x),
    }
}

fn apply_transform(ctx: &mut Context, x: ExprId, transform: Transform) -> ExprId {
    match transform {
        Transform::XOverSqrt(base) => {
            let b = build_base(ctx, x, base);
            let sqrt_b = build_sqrt(ctx, b);
            ctx.add(Expr::Div(x, sqrt_b))
        }
        Transform::SqrtOverX(base) => {
            let b = build_base(ctx, x, base);
            let sqrt_b = build_sqrt(ctx, b);
            ctx.add(Expr::Div(sqrt_b, x))
        }
        Transform::OneOverSqrt(base) => {
            let one = ctx.num(1);
            let b = build_base(ctx, x, base);
            let sqrt_b = build_sqrt(ctx, b);
            ctx.add(Expr::Div(one, sqrt_b))
        }
        Transform::OneOverX => {
            let one = ctx.num(1);
            ctx.add(Expr::Div(one, x))
        }
        Transform::JustSqrt(base) => {
            let b = build_base(ctx, x, base);
            build_sqrt(ctx, b)
        }
    }
}

/// Maps (outer_func, inner_func) -> (Transform, description)
const EXPANSIONS: &[(&str, &[&str], Transform, &str)] = &[
    // sin(arctan(x)) → x/√(1+x²)
    (
        "sin",
        &["arctan", "atan"],
        Transform::XOverSqrt(Base::OnePlus),
        "sin(arctan(x)) → x/√(1+x²)",
    ),
    // cos(arctan(x)) → 1/√(1+x²)
    (
        "cos",
        &["arctan", "atan"],
        Transform::OneOverSqrt(Base::OnePlus),
        "cos(arctan(x)) → 1/√(1+x²)",
    ),
    // tan(arcsin(x)) → x/√(1-x²)
    (
        "tan",
        &["arcsin", "asin"],
        Transform::XOverSqrt(Base::OneMinus),
        "tan(arcsin(x)) → x/√(1-x²)",
    ),
    // cot(arcsin(x)) → √(1-x²)/x
    (
        "cot",
        &["arcsin", "asin"],
        Transform::SqrtOverX(Base::OneMinus),
        "cot(arcsin(x)) → √(1-x²)/x",
    ),
    // cos(arcsin(x)) → √(1-x²)
    (
        "cos",
        &["arcsin", "asin"],
        Transform::JustSqrt(Base::OneMinus),
        "cos(arcsin(x)) → √(1-x²)",
    ),
    // sin(arccos(x)) → √(1-x²)
    (
        "sin",
        &["arccos", "acos"],
        Transform::JustSqrt(Base::OneMinus),
        "sin(arccos(x)) → √(1-x²)",
    ),
    // sin(arcsec(x)) → √(x²-1)/x
    (
        "sin",
        &["arcsec", "asec"],
        Transform::SqrtOverX(Base::XMinus),
        "sin(arcsec(x)) → √(x²-1)/x",
    ),
    // cos(arcsec(x)) → 1/x
    (
        "cos",
        &["arcsec", "asec"],
        Transform::OneOverX,
        "cos(arcsec(x)) → 1/x",
    ),
    // tan(arccos(x)) → √(1-x²)/x
    (
        "tan",
        &["arccos", "acos"],
        Transform::SqrtOverX(Base::OneMinus),
        "tan(arccos(x)) → √(1-x²)/x",
    ),
    // cot(arccos(x)) → x/√(1-x²)
    (
        "cot",
        &["arccos", "acos"],
        Transform::XOverSqrt(Base::OneMinus),
        "cot(arccos(x)) → x/√(1-x²)",
    ),
    // sec(arctan(x)) → √(1+x²)
    (
        "sec",
        &["arctan", "atan"],
        Transform::JustSqrt(Base::OnePlus),
        "sec(arctan(x)) → √(1+x²)",
    ),
    // csc(arctan(x)) → √(1+x²)/x
    (
        "csc",
        &["arctan", "atan"],
        Transform::SqrtOverX(Base::OnePlus),
        "csc(arctan(x)) → √(1+x²)/x",
    ),
    // sec(arcsin(x)) → 1/√(1-x²)
    (
        "sec",
        &["arcsin", "asin"],
        Transform::OneOverSqrt(Base::OneMinus),
        "sec(arcsin(x)) → 1/√(1-x²)",
    ),
    // csc(arcsin(x)) → 1/x
    (
        "csc",
        &["arcsin", "asin"],
        Transform::OneOverX,
        "csc(arcsin(x)) → 1/x",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn expands_sin_arctan() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("x");
        let (expanded, desc) =
            expand_trig_inverse_composition(&mut ctx, "sin", "arctan", x).expect("expanded");
        assert_eq!(desc, "sin(arctan(x)) → x/√(1+x²)");
        let expected = parse("x/((x^2+1)^(1/2))", &mut ctx).expect("expected");
        let checker = crate::semantic_equality::SemanticEqualityChecker::new(&ctx);
        assert!(checker.are_equal(expanded, expected));
    }

    #[test]
    fn expands_cos_arcsec() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("x");
        let (expanded, desc) =
            expand_trig_inverse_composition(&mut ctx, "cos", "arcsec", x).expect("expanded");
        assert_eq!(desc, "cos(arcsec(x)) → 1/x");
        let expected = parse("1/x", &mut ctx).expect("expected");
        let checker = crate::semantic_equality::SemanticEqualityChecker::new(&ctx);
        assert!(checker.are_equal(expanded, expected));
    }

    #[test]
    fn returns_none_for_unknown_combo() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("x");
        let expanded = expand_trig_inverse_composition(&mut ctx, "sin", "arcsin", x);
        assert!(expanded.is_none());
    }
}
