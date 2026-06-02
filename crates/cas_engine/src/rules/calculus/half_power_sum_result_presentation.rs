use super::polynomial_support::polynomial_radicand_for_calculus_presentation;
use super::scalar_presentation::scale_expr_for_calculus_presentation;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

fn half_power_term_for_integration_presentation(
    ctx: &Context,
    term: ExprId,
    sign: cas_math::expr_nary::Sign,
    var_name: &str,
) -> Option<(ExprId, BigRational, u32)> {
    let mut coefficient = BigRational::one();
    let mut base_and_offset = None;

    for factor in cas_math::expr_nary::MulView::from_expr(ctx, term).factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            coefficient *= value;
            continue;
        }

        let (base, power) = match ctx.get(factor) {
            Expr::Function(fn_id, args)
                if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) =>
            {
                (args[0], BigRational::new(1.into(), 2.into()))
            }
            Expr::Pow(base, exp) => {
                let power = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
                (*base, power)
            }
            _ => return None,
        };

        polynomial_radicand_for_calculus_presentation(ctx, base, var_name)?;
        let offset = power - BigRational::new(1.into(), 2.into());
        if !offset.is_integer() || offset.is_negative() {
            return None;
        }
        let offset = offset.to_integer().to_u32()?;
        if offset > 4 || base_and_offset.replace((base, offset)).is_some() {
            return None;
        }
    }

    if sign == cas_math::expr_nary::Sign::Neg {
        coefficient = -coefficient;
    }

    let (base, offset) = base_and_offset?;
    (!coefficient.is_zero()).then_some((base, coefficient, offset))
}

pub(super) fn compact_half_power_sum_root_product_for_integration_presentation(
    ctx: &mut Context,
    result: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let terms = cas_math::expr_nary::AddView::from_expr(ctx, result).terms;
    if terms.len() < 2 || terms.len() > 4 {
        return None;
    }

    let mut common_base = None;
    let mut polynomial_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let (base, coefficient, offset) =
            half_power_term_for_integration_presentation(ctx, term, sign, var_name)?;
        if let Some(existing) = common_base {
            if compare_expr(ctx, existing, base) != std::cmp::Ordering::Equal {
                return None;
            }
        } else {
            common_base = Some(base);
        }

        let power_free = match offset {
            0 => ctx.num(1),
            1 => base,
            _ => {
                let exp = ctx.num(offset as i64);
                ctx.add(Expr::Pow(base, exp))
            }
        };
        polynomial_terms.push(scale_expr_for_calculus_presentation(
            ctx,
            coefficient,
            power_free,
        ));
    }

    let base = common_base?;
    let raw_polynomial = cas_math::expr_nary::build_balanced_add(ctx, &polynomial_terms);
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };
    let polynomial = multipoly_from_expr(ctx, raw_polynomial, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw_polynomial);
    let sqrt_base = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    Some(cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[sqrt_base, polynomial],
    ))
}
