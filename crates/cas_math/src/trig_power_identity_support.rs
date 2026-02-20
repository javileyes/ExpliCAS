use crate::expr_nary::mul_leaves;
use crate::numeric_eval::as_rational_const;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use std::cmp::Ordering;

fn extract_trig_pow_n(ctx: &Context, term: ExprId, n: i64) -> Option<(ExprId, &'static str)> {
    if let Expr::Pow(base, exp) = ctx.get(term) {
        if let Expr::Number(pow) = ctx.get(*exp) {
            if pow.is_integer() && *pow.numer() == n.into() {
                if let Expr::Function(fn_id, args) = ctx.get(*base) {
                    if args.len() == 1 {
                        match ctx.builtin_of(*fn_id) {
                            Some(BuiltinFn::Sin) => return Some((args[0], "sin")),
                            Some(BuiltinFn::Cos) => return Some((args[0], "cos")),
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    None
}

pub fn extract_trig_pow2(ctx: &Context, term: ExprId) -> Option<(ExprId, &'static str)> {
    extract_trig_pow_n(ctx, term, 2)
}

pub fn extract_trig_pow4(ctx: &Context, term: ExprId) -> Option<(ExprId, &'static str)> {
    extract_trig_pow_n(ctx, term, 4)
}

pub fn extract_trig_pow6(ctx: &Context, term: ExprId) -> Option<(ExprId, &'static str)> {
    extract_trig_pow_n(ctx, term, 6)
}

/// Extract `coeff * sin(arg)^2 * cos(arg)^2` from a product term.
/// Returns `(coeff, arg)` when both squared trig factors share the same argument.
pub fn extract_sin2_cos2_product(ctx: &mut Context, term: ExprId) -> Option<(ExprId, ExprId)> {
    let factors = mul_leaves(ctx, term);
    if factors.len() < 2 {
        return None;
    }

    let mut sin2_arg: Option<ExprId> = None;
    let mut cos2_arg: Option<ExprId> = None;
    let mut other_factors: Vec<ExprId> = Vec::new();

    for factor in factors {
        if let Some((arg, name)) = extract_trig_pow2(ctx, factor) {
            match name {
                "sin" if sin2_arg.is_none() => sin2_arg = Some(arg),
                "cos" if cos2_arg.is_none() => cos2_arg = Some(arg),
                _ => other_factors.push(factor),
            }
        } else {
            other_factors.push(factor);
        }
    }

    let sin_arg = sin2_arg?;
    let cos_arg = cos2_arg?;
    if cas_ast::ordering::compare_expr(ctx, sin_arg, cos_arg) != Ordering::Equal {
        return None;
    }

    let coeff = if other_factors.is_empty() {
        ctx.num(1)
    } else if other_factors.len() == 1 {
        other_factors[0]
    } else {
        let mut acc = other_factors[0];
        for factor in &other_factors[1..] {
            acc = ctx.add(Expr::Mul(acc, *factor));
        }
        acc
    };

    Some((coeff, sin_arg))
}

/// Check if a coefficient expression is exactly 3.
pub fn coeff_is_three(ctx: &mut Context, coeff: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(coeff) {
        return n.is_integer() && *n.numer() == 3.into();
    }
    if let Some(v) = as_rational_const(ctx, coeff) {
        return v == num_rational::BigRational::from_integer(3.into());
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn extract_pow_variants_detect_sin_and_cos() {
        let mut ctx = Context::new();
        let s2 = parse("sin(x)^2", &mut ctx).expect("s2");
        let c4 = parse("cos(y)^4", &mut ctx).expect("c4");
        let s6 = parse("sin(z)^6", &mut ctx).expect("s6");

        assert!(extract_trig_pow2(&ctx, s2).is_some());
        assert!(extract_trig_pow4(&ctx, c4).is_some());
        assert!(extract_trig_pow6(&ctx, s6).is_some());
    }

    #[test]
    fn extract_sin2_cos2_product_finds_coeff_and_arg() {
        let mut ctx = Context::new();
        let term = parse("3*sin(t)^2*cos(t)^2", &mut ctx).expect("term");
        let (coeff, arg) = extract_sin2_cos2_product(&mut ctx, term).expect("match");
        let three = parse("3", &mut ctx).expect("three");
        let t = parse("t", &mut ctx).expect("t");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, coeff, three),
            Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, t),
            Ordering::Equal
        );
        assert!(coeff_is_three(&mut ctx, coeff));
    }
}
