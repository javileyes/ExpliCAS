use crate::expr_extract::extract_exp_argument;
use crate::expr_relations::is_negation;
use cas_ast::{Context, Expr, ExprId};

/// Try to extract the pair `(exp(a), exp(-a))` from an additive expression.
///
/// Returns `Some((arg, is_cosh_pattern, positive_first))` where:
/// - `arg`: the canonical positive argument `a`
/// - `is_cosh_pattern`: `true` for `exp(a) + exp(-a)`, `false` for difference forms
/// - `positive_first`: for difference forms, whether `exp(a)` appears before `exp(-a)`
pub fn extract_exp_pair(ctx: &Context, id: ExprId) -> Option<(ExprId, bool, bool)> {
    // Add: exp(a) + exp(-a) -> cosh(a)
    if let Expr::Add(l, r) = ctx.get(id) {
        if let (Some(l_arg), Some(r_arg)) =
            (extract_exp_argument(ctx, *l), extract_exp_argument(ctx, *r))
        {
            if is_negation(ctx, r_arg, l_arg) {
                return Some((l_arg, true, true));
            }
            if is_negation(ctx, l_arg, r_arg) {
                return Some((r_arg, true, true));
            }
        }
    }

    // Sub: exp(a) - exp(-a) / exp(-a) - exp(a) -> +/- sinh(a)
    if let Expr::Sub(l, r) = ctx.get(id) {
        if let (Some(l_arg), Some(r_arg)) =
            (extract_exp_argument(ctx, *l), extract_exp_argument(ctx, *r))
        {
            if is_negation(ctx, r_arg, l_arg) {
                return Some((l_arg, false, true));
            }
            if is_negation(ctx, l_arg, r_arg) {
                return Some((r_arg, false, false));
            }
        }
    }

    // Add with explicit negation: Add(exp(a), Neg(exp(-a))) and symmetric forms.
    if let Expr::Add(l, r) = ctx.get(id) {
        if let Expr::Neg(neg_inner) = ctx.get(*r) {
            if let (Some(l_arg), Some(r_arg)) = (
                extract_exp_argument(ctx, *l),
                extract_exp_argument(ctx, *neg_inner),
            ) {
                if is_negation(ctx, r_arg, l_arg) {
                    return Some((l_arg, false, true));
                }
                if is_negation(ctx, l_arg, r_arg) {
                    return Some((r_arg, false, false));
                }
            }
        }
        if let Expr::Neg(neg_inner) = ctx.get(*l) {
            if let (Some(l_arg), Some(r_arg)) = (
                extract_exp_argument(ctx, *neg_inner),
                extract_exp_argument(ctx, *r),
            ) {
                if is_negation(ctx, l_arg, r_arg) {
                    return Some((r_arg, false, true));
                }
                if is_negation(ctx, r_arg, l_arg) {
                    return Some((l_arg, false, false));
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::extract_exp_pair;
    use cas_ast::{BuiltinFn, Context, Expr};

    fn exp(ctx: &mut Context, arg: cas_ast::ExprId) -> cas_ast::ExprId {
        ctx.call_builtin(BuiltinFn::Exp, vec![arg])
    }

    #[test]
    fn extract_exp_pair_detects_cosh_pattern() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let nx = ctx.add(Expr::Neg(x));
        let exp_x = exp(&mut ctx, x);
        let exp_nx = exp(&mut ctx, nx);
        let expr = ctx.add(Expr::Add(exp_x, exp_nx));

        let (arg, is_cosh, positive_first) = extract_exp_pair(&ctx, expr).expect("pair");
        assert_eq!(arg, x);
        assert!(is_cosh);
        assert!(positive_first);
    }

    #[test]
    fn extract_exp_pair_detects_sinh_sub_patterns() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let nx = ctx.add(Expr::Neg(x));
        let exp_x = exp(&mut ctx, x);
        let exp_nx = exp(&mut ctx, nx);
        let pos_first = ctx.add(Expr::Sub(exp_x, exp_nx));
        let neg_first = ctx.add(Expr::Sub(exp_nx, exp_x));

        let (_, is_cosh1, positive_first1) = extract_exp_pair(&ctx, pos_first).expect("pair");
        assert!(!is_cosh1);
        assert!(positive_first1);

        let (_, is_cosh2, _) = extract_exp_pair(&ctx, neg_first).expect("pair");
        assert!(!is_cosh2);
    }
}
