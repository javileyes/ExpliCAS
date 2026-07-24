//! Rewrite helpers for algebraic constants (currently focused on `phi`).

use crate::expr_predicates::{is_half_expr, is_one_expr};
use crate::number_theory_support::is_sqrt_of_integer_expr;
use cas_ast::{Constant, Context, Expr, ExprId};
use num_rational::BigRational;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantRewrite {
    pub rewritten: ExprId,
    pub kind: ConstantRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstantRewriteKind {
    RecognizePhi,
    PhiSquared,
    PhiReciprocal,
    PhiLinearSurd,
}

/// Canonicalize a CONSTANT additive expression that mixes `phi` with rationals
/// and/or `sqrt(5)` into its exact `A + B·√5` form via the quadratic-surd
/// walker (which knows `phi = 1/2 + (1/2)·√5`). Fires only when the surd form
/// is a genuine reduction, as declared intention (not a display preference):
/// - the value is exactly RATIONAL (`sqrt(5) + 1 - 2·phi → 0`), or
/// - the rational part vanishes (`2·phi - 1 → sqrt(5)`), or
/// - the expression mixes `phi` with an EXPLICIT √-term (single-basis rewrite).
///
/// A lone `phi`, and phi-linear forms whose surd spelling is no simpler
/// (`1 + phi`), are left alone — `phi` stays the display canonical there.
pub fn try_rewrite_phi_linear_surd_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ConstantRewrite> {
    use num_traits::{One, Zero};

    fn contains_phi(ctx: &Context, e: ExprId) -> bool {
        match ctx.get(e) {
            Expr::Constant(Constant::Phi) => true,
            Expr::Neg(i) => contains_phi(ctx, *i),
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                contains_phi(ctx, *l) || contains_phi(ctx, *r)
            }
            _ => false,
        }
    }
    fn contains_explicit_sqrt(ctx: &Context, e: ExprId) -> bool {
        match ctx.get(e) {
            Expr::Function(fn_id, args) => args.len() == 1 && ctx.sym_name(*fn_id) == "sqrt",
            Expr::Neg(i) => contains_explicit_sqrt(ctx, *i),
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                contains_explicit_sqrt(ctx, *l) || contains_explicit_sqrt(ctx, *r)
            }
            Expr::Pow(b, _) => contains_explicit_sqrt(ctx, *b),
            _ => false,
        }
    }

    // Only additive/negated roots: a lone `phi` (or a product) is not this
    // rule's business, and the walker below already declines variables.
    if !matches!(
        ctx.get(expr),
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_)
    ) {
        return None;
    }
    if !contains_phi(ctx, expr) {
        return None;
    }
    let (a, b, n) = crate::root_forms::as_linear_surd(ctx, expr)?;

    let rational_result = b.is_zero();
    let pure_surd = a.is_zero() && !b.is_zero();
    let mixed_basis = contains_explicit_sqrt(ctx, expr);
    if !(rational_result || pure_surd || mixed_basis) {
        return None;
    }

    let rebuilt = if rational_result {
        ctx.add(Expr::Number(a))
    } else {
        let n_expr = ctx.add(Expr::Number(n));
        let sqrt_n = ctx.call("sqrt", vec![n_expr]);
        let surd_term = if b.is_one() {
            sqrt_n
        } else if (-&b).is_one() {
            ctx.add(Expr::Neg(sqrt_n))
        } else {
            let b_expr = ctx.add(Expr::Number(b));
            ctx.add(Expr::Mul(b_expr, sqrt_n))
        };
        if a.is_zero() {
            surd_term
        } else {
            let a_expr = ctx.add(Expr::Number(a));
            ctx.add(Expr::Add(a_expr, surd_term))
        }
    };
    if rebuilt == expr {
        return None;
    }
    Some(ConstantRewrite {
        rewritten: rebuilt,
        kind: ConstantRewriteKind::PhiLinearSurd,
    })
}

/// Recognize `(1 + sqrt(5))/2` in division or product-with-half form.
pub fn try_rewrite_recognize_phi_expr(ctx: &mut Context, expr: ExprId) -> Option<ConstantRewrite> {
    // Pattern 1: (1 + sqrt(5)) / 2
    if let Expr::Div(num, den) = ctx.get(expr) {
        if !matches!(ctx.get(*den), Expr::Number(d) if *d == BigRational::from_integer(2.into())) {
            return None;
        }

        if let Expr::Add(l, r) = ctx.get(*num) {
            let matches_phi_sum = (is_one_expr(ctx, *l) && is_sqrt_of_integer_expr(ctx, *r, 5))
                || (is_sqrt_of_integer_expr(ctx, *l, 5) && is_one_expr(ctx, *r));
            if matches_phi_sum {
                return Some(ConstantRewrite {
                    rewritten: ctx.add(Expr::Constant(Constant::Phi)),
                    kind: ConstantRewriteKind::RecognizePhi,
                });
            }
        }
    }

    // Pattern 2: (1/2) * (1 + sqrt(5))
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let sum_id = if is_half_expr(ctx, *l) {
            *r
        } else if is_half_expr(ctx, *r) {
            *l
        } else {
            return None;
        };

        if let Expr::Add(a, b) = ctx.get(sum_id) {
            let matches_phi_sum = (is_one_expr(ctx, *a) && is_sqrt_of_integer_expr(ctx, *b, 5))
                || (is_sqrt_of_integer_expr(ctx, *a, 5) && is_one_expr(ctx, *b));
            if matches_phi_sum {
                return Some(ConstantRewrite {
                    rewritten: ctx.add(Expr::Constant(Constant::Phi)),
                    kind: ConstantRewriteKind::RecognizePhi,
                });
            }
        }
    }

    None
}

/// Rewrite `phi^2 -> phi + 1`.
pub fn try_rewrite_phi_squared_expr(ctx: &mut Context, expr: ExprId) -> Option<ConstantRewrite> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !matches!(ctx.get(*base), Expr::Constant(Constant::Phi)) {
        return None;
    }
    if !matches!(ctx.get(*exp), Expr::Number(n) if *n == BigRational::from_integer(2.into())) {
        return None;
    }

    let phi = ctx.add(Expr::Constant(Constant::Phi));
    let one = ctx.num(1);
    Some(ConstantRewrite {
        rewritten: ctx.add(Expr::Add(phi, one)),
        kind: ConstantRewriteKind::PhiSquared,
    })
}

/// Rewrite `1/phi -> phi - 1`.
pub fn try_rewrite_phi_reciprocal_expr(ctx: &mut Context, expr: ExprId) -> Option<ConstantRewrite> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    if !is_one_expr(ctx, *num) {
        return None;
    }
    if !matches!(ctx.get(*den), Expr::Constant(Constant::Phi)) {
        return None;
    }

    let phi = ctx.add(Expr::Constant(Constant::Phi));
    let one = ctx.num(1);
    let neg_one = ctx.add(Expr::Neg(one));
    Some(ConstantRewrite {
        rewritten: ctx.add(Expr::Add(phi, neg_one)),
        kind: ConstantRewriteKind::PhiReciprocal,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_phi_reciprocal_expr, try_rewrite_phi_squared_expr,
        try_rewrite_recognize_phi_expr,
    };
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn recognizes_phi_division_form() {
        let mut ctx = Context::new();
        let expr = parse("(1 + sqrt(5)) / 2", &mut ctx).expect("parse");
        let rewrite = try_rewrite_recognize_phi_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rendered(&ctx, rewrite.rewritten), "phi");
    }

    #[test]
    fn rewrites_phi_squared() {
        let mut ctx = Context::new();
        let expr = parse("phi^2", &mut ctx).expect("parse");
        let rewrite = try_rewrite_phi_squared_expr(&mut ctx, expr).expect("rewrite");
        let text = rendered(&ctx, rewrite.rewritten);
        assert!(text.contains("phi"));
        assert!(text.contains("1"));
    }

    #[test]
    fn rewrites_phi_reciprocal() {
        let mut ctx = Context::new();
        let expr = parse("1/phi", &mut ctx).expect("parse");
        let rewrite = try_rewrite_phi_reciprocal_expr(&mut ctx, expr).expect("rewrite");
        assert!(rendered(&ctx, rewrite.rewritten).contains("phi"));
    }
}
