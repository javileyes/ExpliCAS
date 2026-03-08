use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;

pub(super) fn extract_sum_three_cubes_bases(
    context: &Context,
    expr: ExprId,
) -> Option<[ExprId; 3]> {
    let terms = cas_math::expr_nary::add_leaves(context, expr);
    if terms.len() != 3 {
        return None;
    }

    let mut bases = Vec::with_capacity(3);
    for &term in &terms {
        bases.push(extract_cube_base(context, term)?);
    }

    Some([bases[0], bases[1], bases[2]])
}

fn extract_cube_base(context: &Context, expr: ExprId) -> Option<ExprId> {
    match context.get(expr).clone() {
        Expr::Pow(base, exp) if is_cube_exponent(context, exp) => Some(base),
        Expr::Neg(inner) => match context.get(inner).clone() {
            Expr::Pow(_, exp) if is_cube_exponent(context, exp) => Some(inner),
            _ => None,
        },
        _ => None,
    }
}

fn is_cube_exponent(context: &Context, expr: ExprId) -> bool {
    match context.get(expr).clone() {
        Expr::Number(number) => number.is_integer() && number.to_integer() == BigInt::from(3),
        _ => false,
    }
}
