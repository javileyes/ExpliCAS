//! Support helpers for integration-preparation rewrite rules.

use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_extract::extract_i64_multiplier_and_base;
use cas_math::trig_roots_flatten::flatten_mul_chain;
use num_bigint::BigInt;
use num_rational::BigRational;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CosProductTelescopingRewrite {
    pub rewritten: ExprId,
    pub assume_nonzero_expr: ExprId,
    pub desc: String,
}

/// Try Morrie's-law telescoping:
/// `cos(u)*cos(2u)*...*cos(2^(n-1)u) -> sin(2^n u)/(2^n sin(u))`.
pub fn try_rewrite_cos_product_telescoping_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CosProductTelescopingRewrite> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut cos_info: Vec<(i64, ExprId)> = Vec::new();
    for &factor in &factors {
        if let Expr::Function(fn_id, args) = ctx.get(factor) {
            if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Cos) && args.len() == 1 {
                let (k, u) = extract_i64_multiplier_and_base(ctx, args[0]);
                cos_info.push((k, u));
            }
        }
    }
    if cos_info.len() < 2 {
        return None;
    }

    let base_u = cos_info[0].1;
    let mut multipliers = Vec::new();
    for (k, u) in &cos_info {
        if *u != base_u {
            return None;
        }
        multipliers.push(*k);
    }

    multipliers.sort();
    let base_mult = multipliers[0];
    if base_mult <= 0 {
        return None;
    }

    let n = multipliers.len();
    for (i, &m) in multipliers.iter().enumerate() {
        let expected = base_mult * (1i64 << i);
        if m != expected {
            return None;
        }
    }

    let power_of_2 = 1i64 << n;
    let final_mult = base_mult * power_of_2;

    let final_mult_num = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(
        final_mult,
    ))));
    let final_arg = ctx.add(Expr::Mul(final_mult_num, base_u));

    let base_mult_num = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(
        base_mult,
    ))));
    let base_arg = if base_mult == 1 {
        base_u
    } else {
        ctx.add(Expr::Mul(base_mult_num, base_u))
    };

    let sin_num = ctx.call_builtin(BuiltinFn::Sin, vec![final_arg]);
    let sin_den = ctx.call_builtin(BuiltinFn::Sin, vec![base_arg]);

    let power_of_2_num = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(
        power_of_2,
    ))));
    let denominator = ctx.add(Expr::Mul(power_of_2_num, sin_den));
    let rewritten = ctx.add(Expr::Div(sin_num, denominator));

    Some(CosProductTelescopingRewrite {
        rewritten,
        assume_nonzero_expr: sin_den,
        desc: format!(
            "cos telescoping (Morrie's law): cos(u)·cos(2u)·...·cos(2^{}u) → sin(2^{}u)/(2^{}·sin(u))",
            n - 1,
            n,
            n
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_cos_product_telescoping_expr;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn rewrites_morrie_telescoping_product() {
        let mut ctx = Context::new();
        let expr = parse("cos(x)*cos(2*x)*cos(4*x)", &mut ctx).unwrap_or_else(|_| panic!("parse"));
        let rewrite = try_rewrite_cos_product_telescoping_expr(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));
        let text = rendered(&ctx, rewrite.rewritten);
        assert!(text.contains("sin(8 * x)"));
        assert!(text.contains("8"));
        assert!(text.contains("sin(x)"));
    }

    #[test]
    fn rejects_non_dyadic_multiplier_pattern() {
        let mut ctx = Context::new();
        let expr = parse("cos(x)*cos(3*x)", &mut ctx).unwrap_or_else(|_| panic!("parse"));
        assert!(try_rewrite_cos_product_telescoping_expr(&mut ctx, expr).is_none());
    }
}
