//! Helpers for zero-numerator fraction canonicalization.

use cas_ast::{collect_variables, Context, Expr, ExprId};

/// Detect whether `numerator` simplifies to zero under a two-pass expansion.
pub fn numerator_simplifies_to_zero_with<FExpand>(
    ctx: &mut Context,
    numerator: ExprId,
    mut expand: FExpand,
) -> bool
where
    FExpand: FnMut(&mut Context, ExprId) -> ExprId,
{
    let pass1 = expand(ctx, numerator);
    let pass2 = expand(ctx, pass1);
    if crate::numeric_eval::numeric_poly_zero_check(ctx, pass2) {
        return true;
    }

    let zero = ctx.num(0);
    crate::poly_compare::poly_eq(ctx, pass2, zero)
}

/// Build canonical `0` or `0/den` expression depending on variable presence in denominator.
///
/// Returns:
/// - `0` when denominator has no variables.
/// - `0/den` when denominator depends on variables (preserves domain restrictions).
pub fn build_zero_or_zero_over_den(ctx: &mut Context, den: ExprId) -> ExprId {
    let zero = ctx.num(0);
    let den_vars = collect_variables(ctx, den);
    if den_vars.is_empty() {
        zero
    } else {
        ctx.add(Expr::Div(zero, den))
    }
}

#[cfg(test)]
mod tests {
    use super::{build_zero_or_zero_over_den, numerator_simplifies_to_zero_with};
    use cas_ast::{Context, Expr};
    use cas_parser::parse;

    #[test]
    fn detects_zero_after_expand() {
        let mut ctx = Context::new();
        let num = parse("(x+1)*(x-1) - (x^2-1)", &mut ctx).expect("parse");
        assert!(numerator_simplifies_to_zero_with(
            &mut ctx,
            num,
            crate::expand_ops::expand
        ));
    }

    #[test]
    fn detects_zero_via_poly_eq_without_needing_expand() {
        let mut ctx = Context::new();
        let num = parse("u*(u+1) - (u^2 + u)", &mut ctx).expect("parse");
        assert!(numerator_simplifies_to_zero_with(&mut ctx, num, |_, e| e));
    }

    #[test]
    fn builds_plain_zero_for_numeric_denominator() {
        let mut ctx = Context::new();
        let den = parse("6", &mut ctx).expect("parse");
        let expr = build_zero_or_zero_over_den(&mut ctx, den);
        assert!(matches!(ctx.get(expr), Expr::Number(_)));
    }

    #[test]
    fn builds_zero_fraction_for_symbolic_denominator() {
        let mut ctx = Context::new();
        let den = parse("x+1", &mut ctx).expect("parse");
        let expr = build_zero_or_zero_over_den(&mut ctx, den);
        assert!(matches!(ctx.get(expr), Expr::Div(_, _)));
    }
}
