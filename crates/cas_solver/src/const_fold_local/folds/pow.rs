use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::Signed;

fn literal_rat(ctx: &Context, id: ExprId) -> Option<BigRational> {
    match ctx.get(id) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Neg(inner) => {
            if let Expr::Number(n) = ctx.get(*inner) {
                Some(-n.clone())
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(crate) fn fold_pow(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    value_domain: crate::ValueDomain,
    _branch: crate::BranchPolicy,
) -> Option<ExprId> {
    if let Some(result) = cas_math::const_eval::try_eval_pow_literal(ctx, base, exp) {
        return Some(result);
    }

    let base_rat = literal_rat(ctx, base)?;
    let exp_rat = literal_rat(ctx, exp)?;

    let exp_rat = if exp_rat.denom().is_negative() {
        BigRational::new(-exp_rat.numer().clone(), -exp_rat.denom().clone())
    } else {
        exp_rat
    };

    if exp_rat == BigRational::new(1.into(), 2.into())
        && base_rat == BigRational::from_integer((-1).into())
    {
        return match value_domain {
            crate::ValueDomain::RealOnly => {
                Some(ctx.add(Expr::Constant(cas_ast::Constant::Undefined)))
            }
            crate::ValueDomain::ComplexEnabled => {
                Some(ctx.add(Expr::Constant(cas_ast::Constant::I)))
            }
        };
    }

    None
}
