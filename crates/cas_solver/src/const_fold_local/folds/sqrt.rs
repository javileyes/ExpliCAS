use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

pub(crate) fn fold_sqrt(
    ctx: &mut Context,
    base: ExprId,
    value_domain: crate::ValueDomain,
) -> Option<ExprId> {
    let n = match ctx.get(base) {
        Expr::Number(n) => n.clone(),
        _ => return None,
    };

    if n.is_negative() {
        match value_domain {
            crate::ValueDomain::RealOnly => {
                Some(ctx.add(Expr::Constant(cas_ast::Constant::Undefined)))
            }
            crate::ValueDomain::ComplexEnabled => {
                let pos_n = -n;
                let pos_n_expr = ctx.add(Expr::Number(pos_n));
                let sqrt_pos = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![pos_n_expr]);
                let i = ctx.add(Expr::Constant(cas_ast::Constant::I));
                Some(ctx.add(Expr::Mul(i, sqrt_pos)))
            }
        }
    } else if n.is_zero() {
        Some(ctx.num(0))
    } else {
        try_exact_sqrt(ctx, &n)
    }
}

fn try_exact_sqrt(ctx: &mut Context, n: &BigRational) -> Option<ExprId> {
    if !n.is_integer() {
        return None;
    }
    let num = n.numer();
    let sqrt_num = num.sqrt();
    if &(&sqrt_num * &sqrt_num) == num {
        Some(ctx.add(Expr::Number(BigRational::from_integer(sqrt_num))))
    } else {
        None
    }
}
