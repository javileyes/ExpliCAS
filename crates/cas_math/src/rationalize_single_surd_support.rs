//! Support for light rationalization of a single numeric surd denominator.

use crate::build::mul2_raw;
use crate::expr_destructure::as_div;
use crate::root_forms::extract_numeric_sqrt_radicand;
use cas_ast::{count_nodes, Context, Expr, ExprId};

/// Rewrite payload for single-surd denominator rationalization.
#[derive(Debug, Clone, Copy)]
pub struct RationalizeSingleSurdRewrite {
    pub rewritten: ExprId,
    pub num: ExprId,
    pub den: ExprId,
    pub new_num: ExprId,
    pub new_den: ExprId,
}

/// Try to rewrite `num / (k*sqrt(n))` into `(num*sqrt(n)) / (k*n)`.
///
/// Handles shallow denominator forms:
/// - `sqrt(n)`
/// - `k * sqrt(n)` or `sqrt(n) * k`
pub fn try_rewrite_rationalize_single_surd_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RationalizeSingleSurdRewrite> {
    let (num, den) = as_div(ctx, expr)?;

    let (sqrt_n_value, other_den_factors): (i64, Vec<ExprId>) = match ctx.get(den) {
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if let Some(n) = extract_numeric_sqrt_radicand(ctx, l) {
                (n, vec![r])
            } else if let Some(n) = extract_numeric_sqrt_radicand(ctx, r) {
                (n, vec![l])
            } else {
                return None;
            }
        }
        _ => (extract_numeric_sqrt_radicand(ctx, den)?, vec![]),
    };

    let n_expr = ctx.num(sqrt_n_value);
    let half = ctx.rational(1, 2);
    let sqrt_n = ctx.add(Expr::Pow(n_expr, half));
    let new_num = mul2_raw(ctx, num, sqrt_n);

    let n_in_den = ctx.num(sqrt_n_value);
    let new_den = if other_den_factors.is_empty() {
        n_in_den
    } else {
        let mut den_product = other_den_factors[0];
        for &f in &other_den_factors[1..] {
            den_product = mul2_raw(ctx, den_product, f);
        }
        mul2_raw(ctx, den_product, n_in_den)
    };

    let rewritten = ctx.add(Expr::Div(new_num, new_den));
    if count_nodes(ctx, rewritten) > count_nodes(ctx, expr) + 10 {
        return None;
    }

    Some(RationalizeSingleSurdRewrite {
        rewritten,
        num,
        den,
        new_num,
        new_den,
    })
}

/// Build a user-facing description for single-surd rationalization.
///
/// Caller provides expression rendering to keep this module independent from
/// formatter crates.
pub fn format_rationalize_single_surd_desc_with<FRender>(
    rewrite: RationalizeSingleSurdRewrite,
    mut render_expr: FRender,
) -> String
where
    FRender: FnMut(ExprId) -> String,
{
    format!(
        "{} / {} -> {} / {}",
        render_expr(rewrite.num),
        render_expr(rewrite.den),
        render_expr(rewrite.new_num),
        render_expr(rewrite.new_den)
    )
}

#[cfg(test)]
mod tests {
    use super::{
        format_rationalize_single_surd_desc_with, try_rewrite_rationalize_single_surd_expr,
    };
    use crate::root_forms::extract_numeric_sqrt_radicand;
    use cas_ast::{Context, Expr};
    use cas_parser::parse;
    use num_rational::BigRational;

    #[test]
    fn rationalizes_single_surd_denominator() {
        let mut ctx = Context::new();
        let expr = parse("x/sqrt(3)", &mut ctx).expect("parse");
        let out = try_rewrite_rationalize_single_surd_expr(&mut ctx, expr).expect("rewrite");
        let Expr::Div(new_num, new_den) = ctx.get(out.rewritten) else {
            panic!("expected division rewrite");
        };
        assert!(
            matches!(ctx.get(*new_den), Expr::Number(n) if *n == BigRational::from_integer(3.into()))
        );
        assert!(contains_sqrt_3_factor(&ctx, *new_num));
    }

    fn contains_sqrt_3_factor(ctx: &Context, expr: cas_ast::ExprId) -> bool {
        if extract_numeric_sqrt_radicand(ctx, expr) == Some(3) {
            return true;
        }
        match ctx.get(expr) {
            Expr::Mul(l, r) => contains_sqrt_3_factor(ctx, *l) || contains_sqrt_3_factor(ctx, *r),
            _ => false,
        }
    }

    #[test]
    fn format_desc_includes_all_parts() {
        let mut ctx = Context::new();
        let expr = parse("x/sqrt(3)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_rationalize_single_surd_expr(&mut ctx, expr).expect("rewrite");
        let desc = format_rationalize_single_surd_desc_with(rewrite, |id| format!("{:?}", id));
        assert!(desc.contains("->"));
    }

    #[test]
    fn formatting_helper_returns_desc_for_rewrite() {
        let mut ctx = Context::new();
        let expr = parse("x/sqrt(3)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_rationalize_single_surd_expr(&mut ctx, expr).expect("rewrite");
        let desc = format_rationalize_single_surd_desc_with(rewrite, |id| format!("{:?}", id));
        assert!(desc.contains("->"));
    }
}
