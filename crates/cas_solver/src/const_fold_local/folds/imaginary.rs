use cas_ast::{Context, Expr, ExprId};

fn is_imaginary_unit(ctx: &Context, id: ExprId) -> bool {
    matches!(ctx.get(id), Expr::Constant(cas_ast::Constant::I))
}

fn is_neg_of_i(ctx: &Context, id: ExprId) -> bool {
    if let Expr::Neg(inner) = ctx.get(id) {
        is_imaginary_unit(ctx, *inner)
    } else {
        false
    }
}

pub(crate) fn fold_mul_imaginary(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    value_domain: crate::ValueDomain,
) -> Option<ExprId> {
    if value_domain != crate::ValueDomain::ComplexEnabled {
        return None;
    }

    if is_imaginary_unit(ctx, a) && is_imaginary_unit(ctx, b) {
        return Some(ctx.num(-1));
    }

    let a_is_neg_i = is_neg_of_i(ctx, a);
    let b_is_neg_i = is_neg_of_i(ctx, b);

    if (a_is_neg_i && is_imaginary_unit(ctx, b)) || (is_imaginary_unit(ctx, a) && b_is_neg_i) {
        return Some(ctx.num(1));
    }

    if a_is_neg_i && b_is_neg_i {
        return Some(ctx.num(-1));
    }

    None
}
