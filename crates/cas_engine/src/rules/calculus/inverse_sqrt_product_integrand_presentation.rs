use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;

pub(super) fn compact_inverse_sqrt_product_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
) -> Option<ExprId> {
    let target = cas_ast::hold::strip_all_holds(ctx, target);

    if let Expr::Pow(base, exp) = ctx.get(target).clone() {
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if exponent == BigRational::new((-1).into(), 2.into()) {
            let radicand_factors = cas_math::expr_nary::mul_leaves(ctx, base);
            if radicand_factors.len() < 2 {
                return None;
            }
            let denominator_factors: Vec<_> = radicand_factors
                .into_iter()
                .map(|factor| ctx.call_builtin(BuiltinFn::Sqrt, vec![factor]))
                .collect();
            let one = ctx.num(1);
            let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
            return Some(ctx.add(Expr::Div(one, denominator)));
        }
    }

    let Expr::Div(num, den) = ctx.get(target).clone() else {
        return None;
    };

    let mut changed = false;
    let mut denominator_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
        let Some(radicand) = extract_square_root_base(ctx, factor) else {
            denominator_factors.push(factor);
            continue;
        };
        let radicand_factors = cas_math::expr_nary::mul_leaves(ctx, radicand);
        if radicand_factors.len() < 2 {
            denominator_factors.push(factor);
            continue;
        }

        changed = true;
        for radicand_factor in radicand_factors {
            denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand_factor]));
        }
    }

    if !changed {
        return None;
    }

    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    Some(ctx.add(Expr::Div(num, denominator)))
}
