//! Additive-term and irrational-shape helpers over AST expressions.

use cas_ast::{BuiltinFn, Context, Expr, ExprId};

/// Build an additive chain from a list of terms.
///
/// Empty input returns `0`.
pub fn build_sum(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    if terms.is_empty() {
        return ctx.num(0);
    }
    let mut result = terms[0];
    for &term in terms.iter().skip(1) {
        result = ctx.add(Expr::Add(result, term));
    }
    result
}

/// Collect additive terms by flattening only `Add(...)` nodes.
///
/// For `a + b + c`, returns `[a, b, c]`.
pub fn collect_additive_terms_flat_add(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    collect_additive_terms_recursive(ctx, expr, &mut terms);
    terms
}

fn collect_additive_terms_recursive(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_additive_terms_recursive(ctx, *left, terms);
            collect_additive_terms_recursive(ctx, *right, terms);
        }
        _ => terms.push(expr),
    }
}

/// Check whether an expression contains irrational-root structure.
///
/// The predicate is structural:
/// - fractional powers are irrational (`x^(1/2)`, etc.)
/// - `sqrt(...)` builtin is irrational
pub fn contains_irrational(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(_, exp) => match ctx.get(*exp) {
            Expr::Number(n) => !n.is_integer(),
            _ => false,
        },
        Expr::Function(name, _) => ctx.is_builtin(*name, BuiltinFn::Sqrt),
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            contains_irrational(ctx, *l) || contains_irrational(ctx, *r)
        }
        Expr::Neg(e) | Expr::Hold(e) => contains_irrational(ctx, *e),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly_compare::poly_eq;
    use cas_ast::Expr;
    use cas_parser::parse;
    use num_rational::BigRational;

    #[test]
    fn build_sum_empty_is_zero() {
        let mut ctx = Context::new();
        let sum = build_sum(&mut ctx, &[]);
        let zero = parse("0", &mut ctx).expect("parse zero");
        assert!(poly_eq(&ctx, sum, zero));
    }

    #[test]
    fn collect_add_terms_flatten_add_tree() {
        let mut ctx = Context::new();
        let expr = parse("a + b + c", &mut ctx).expect("parse");
        let terms = collect_additive_terms_flat_add(&ctx, expr);
        assert_eq!(terms.len(), 3);
    }

    #[test]
    fn contains_irrational_detects_sqrt_and_fractional_pow() {
        let mut ctx = Context::new();
        let sqrt_expr = parse("sqrt(x)", &mut ctx).expect("parse sqrt");
        let x = parse("x", &mut ctx).expect("parse x");
        let one_third = ctx.add(Expr::Number(BigRational::new(1.into(), 3.into())));
        let pow_expr = ctx.add(Expr::Pow(x, one_third));
        let rat_expr = parse("x^2", &mut ctx).expect("parse integer pow");

        assert!(contains_irrational(&ctx, sqrt_expr));
        assert!(contains_irrational(&ctx, pow_expr));
        assert!(!contains_irrational(&ctx, rat_expr));
    }
}
