use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::Ratio;

// ========== Helper Functions ==========

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

// ========== Transform Types ==========

/// How to transform x for the result
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

/// Which polynomial to use inside sqrt
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

// ========== Expansion Table ==========

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

// ========== Unified Rule ==========

define_rule!(
    TrigInverseExpansionRule,
    "Trig of Inverse Trig Expansion",
    Some(vec!["Function"]),
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_args.len() != 1 {
                return None;
            }
            let inner = outer_args[0];

            if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                if inner_args.len() != 1 {
                    return None;
                }
                let x = inner_args[0];

                // Look up in expansion table
                for (outer, inner_variants, transform, description) in EXPANSIONS {
                    if outer_name == *outer && inner_variants.contains(&inner_name.as_str()) {
                        let result = apply_transform(ctx, x, *transform);
                        return Some(Rewrite {
                            new_expr: result,
                            description: description.to_string(),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
            required_conditions: vec![],
                        });
                    }
                }
            }
        }
        None
    }
);

// ========== Registration ==========

pub fn register(simplifier: &mut crate::engine::Simplifier) {
    simplifier.add_rule(Box::new(TrigInverseExpansionRule));
}
