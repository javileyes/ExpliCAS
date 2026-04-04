use cas_ast::ordering::compare_expr;
use cas_ast::views::as_rational_const;
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::expr_nary::add_terms_no_sign;
use num_traits::{One, Zero};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RationalizeRewriteKind {
    RadicalNotableQuotient,
    CancelToZeroAfterRationalize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RationalizeRewrite {
    pub(crate) intermediate: ExprId,
    pub(crate) rewritten: ExprId,
    pub(crate) kind: RationalizeRewriteKind,
}

impl RationalizeRewriteKind {
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::RadicalNotableQuotient => "Polynomial division with opaque substitution",
            Self::CancelToZeroAfterRationalize => "Rationalize linear sqrt denominator",
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::RadicalNotableQuotient => "Polynomial division with opaque substitution",
            Self::CancelToZeroAfterRationalize => "Rationalize Linear Sqrt Denominator",
        }
    }
}

pub(crate) fn try_rewrite_rationalized_target_aware(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<RationalizeRewrite> {
    try_rewrite_radical_notable_quotient_target_aware(ctx, source_expr, target_expr)
        .map(|(intermediate, rewritten)| RationalizeRewrite {
            intermediate,
            rewritten,
            kind: RationalizeRewriteKind::RadicalNotableQuotient,
        })
        .or_else(|| {
            try_rewrite_rationalize_then_cancel_zero_target_aware(ctx, source_expr, target_expr)
                .map(|(intermediate, rewritten)| RationalizeRewrite {
                    intermediate,
                    rewritten,
                    kind: RationalizeRewriteKind::CancelToZeroAfterRationalize,
                })
        })
}

pub(crate) fn looks_rationalizable_source(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Div(_, denominator) => contains_root_like(ctx, *denominator),
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            looks_rationalizable_source(ctx, *left) || looks_rationalizable_source(ctx, *right)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => looks_rationalizable_source(ctx, *inner),
        _ => false,
    }
}

fn contains_root_like(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(_, exp) => matches!(ctx.get(*exp), Expr::Number(n) if !n.is_integer()),
        Expr::Function(name, args)
            if (ctx.is_builtin(*name, BuiltinFn::Sqrt)
                || ctx.is_builtin(*name, BuiltinFn::Root))
                && !args.is_empty() =>
        {
            true
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right) => {
            contains_root_like(ctx, *left) || contains_root_like(ctx, *right)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_root_like(ctx, *inner),
        Expr::Function(_, args) => args.iter().any(|arg| contains_root_like(ctx, *arg)),
        Expr::Matrix { data, .. } => data.iter().any(|arg| contains_root_like(ctx, *arg)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

fn try_rewrite_radical_notable_quotient_target_aware(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let (num, den) = match ctx.get(source_expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    let (root_expr, root_base) = extract_sqrt_minus_one_denominator(ctx, den)?;
    let (num_left, num_right) = match ctx.get(num) {
        Expr::Sub(num_left, num_right) => (*num_left, *num_right),
        _ => return None,
    };
    if !is_one(ctx, num_right) || !matches_root_family_cube(ctx, num_left, root_expr, root_base) {
        return None;
    }

    let one = ctx.num(1);
    let two = ctx.num(2);
    let root_squared = ctx.add(Expr::Pow(root_expr, two));
    if matches_radical_notable_target(ctx, target_expr, one, root_expr, root_base, root_squared) {
        let root_plus_square = ctx.add(Expr::Add(root_expr, root_squared));
        let intermediate = ctx.add(Expr::Add(one, root_plus_square));
        Some((intermediate, target_expr))
    } else {
        None
    }
}

fn extract_sqrt_minus_one_denominator(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return None;
    };
    if !is_one(ctx, *right) {
        return None;
    }
    let root_base = cas_math::root_forms::extract_square_root_base(ctx, *left)?;
    Some((*left, root_base))
}

fn matches_root_family_cube(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    root_expr: ExprId,
    root_base: ExprId,
) -> bool {
    let three = ctx.num(3);
    let root_cubed = ctx.add(Expr::Pow(root_expr, three));
    if compare_expr(ctx, expr, root_cubed) == Ordering::Equal {
        return true;
    }

    let three_halves = num_rational::BigRational::new(3.into(), 2.into());
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if compare_expr(ctx, *base, root_base) == Ordering::Equal
            && as_rational_const(ctx, *exp, 8).is_some_and(|value| value == three_halves)
        {
            return true;
        }
    }

    if let Some(radicand) = cas_math::root_forms::extract_square_root_base(ctx, expr) {
        let three = ctx.num(3);
        let base_cubed = ctx.add(Expr::Pow(root_base, three));
        if compare_expr(ctx, radicand, base_cubed) == Ordering::Equal {
            return true;
        }
    }

    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            (compare_expr(ctx, *left, root_expr) == Ordering::Equal
                && compare_expr(ctx, *right, root_base) == Ordering::Equal)
                || (compare_expr(ctx, *left, root_base) == Ordering::Equal
                    && compare_expr(ctx, *right, root_expr) == Ordering::Equal)
        }
        _ => false,
    }
}

fn matches_radical_notable_target(
    ctx: &mut cas_ast::Context,
    target_expr: ExprId,
    one: ExprId,
    root_expr: ExprId,
    root_base: ExprId,
    root_squared: ExprId,
) -> bool {
    let terms = add_terms_no_sign(ctx, target_expr);
    if terms.len() != 3 {
        return false;
    }

    let mut saw_one = false;
    let mut saw_root = false;
    let mut saw_square_or_base = false;

    for term in terms {
        if compare_expr(ctx, term, one) == Ordering::Equal {
            saw_one = true;
        } else if compare_expr(ctx, term, root_expr) == Ordering::Equal {
            saw_root = true;
        } else if compare_expr(ctx, term, root_base) == Ordering::Equal
            || compare_expr(ctx, term, root_squared) == Ordering::Equal
        {
            saw_square_or_base = true;
        } else {
            return false;
        }
    }

    saw_one && saw_root && saw_square_or_base
}

fn try_rewrite_rationalize_then_cancel_zero_target_aware(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    if !is_zero(ctx, target_expr) {
        return None;
    }

    let (left, right) = match ctx.get(source_expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };

    if let Some((root_expr, root_base)) = extract_unit_over_sqrt_minus_one(ctx, left) {
        if matches_rationalized_conjugate(ctx, right, root_expr, root_base) {
            let intermediate = ctx.add(Expr::Sub(right, right));
            return Some((intermediate, target_expr));
        }
    }

    if let Some((root_expr, root_base)) = extract_unit_over_sqrt_minus_one(ctx, right) {
        if matches_rationalized_conjugate(ctx, left, root_expr, root_base) {
            let intermediate = ctx.add(Expr::Sub(left, left));
            return Some((intermediate, target_expr));
        }
    }

    None
}

fn extract_unit_over_sqrt_minus_one(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    if !is_one(ctx, *num) {
        return None;
    }
    extract_sqrt_minus_one_denominator(ctx, *den)
}

fn matches_rationalized_conjugate(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    root_expr: ExprId,
    root_base: ExprId,
) -> bool {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    matches_one_plus_root(ctx, *num, root_expr)
        && matches_base_minus_one(ctx, *den, root_expr, root_base)
}

fn matches_one_plus_root(ctx: &cas_ast::Context, expr: ExprId, root_expr: ExprId) -> bool {
    let terms = add_terms_no_sign(ctx, expr);
    if terms.len() != 2 {
        return false;
    }

    let mut saw_one = false;
    let mut saw_root = false;
    for term in terms {
        if is_one(ctx, term) {
            saw_one = true;
        } else if compare_expr(ctx, term, root_expr) == Ordering::Equal {
            saw_root = true;
        } else {
            return false;
        }
    }

    saw_one && saw_root
}

fn matches_base_minus_one(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    root_expr: ExprId,
    root_base: ExprId,
) -> bool {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return false,
    };
    if !is_one(ctx, right) {
        return false;
    }

    if compare_expr(ctx, left, root_base) == Ordering::Equal {
        return true;
    }

    let two = ctx.num(2);
    let root_squared = ctx.add(Expr::Pow(root_expr, two));
    compare_expr(ctx, left, root_squared) == Ordering::Equal
}

fn is_one(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

fn is_zero(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_zero())
}

#[cfg(test)]
mod tests {
    use super::{looks_rationalizable_source, try_rewrite_rationalized_target_aware};

    #[test]
    fn detects_tabulated_root_denominators_as_rationalizable() {
        let cases = [
            "1/(sqrt(x)-1)",
            "1/(sqrt(x)+1)",
            "1/(sqrt(x)-2)",
            "1/(sqrt(x)-a)",
            "1/(sqrt(y)-a)",
            "(x^(3/2)-1)/(sqrt(x)-1)",
            "1/(sqrt(x)-1) - (sqrt(x)+1)/(x-1)",
        ];

        for text in cases {
            let mut ctx = cas_ast::Context::new();
            let expr = cas_parser::parse(text, &mut ctx).expect("parse");
            assert!(
                looks_rationalizable_source(&ctx, expr),
                "expected `{text}` to look rationalizable"
            );
        }
    }

    #[test]
    fn rewrites_radical_notable_quotient_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("(x^(3/2)-1)/(sqrt(x)-1)", &mut ctx).expect("source");
        let target = cas_parser::parse("sqrt(x)+x+1", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_rationalized_target_aware(&mut ctx, source, target).expect("rewrite");
        assert_eq!(
            rewrite.kind,
            super::RationalizeRewriteKind::RadicalNotableQuotient
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn rewrites_rationalize_then_cancel_zero_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source =
            cas_parser::parse("1/(sqrt(x)-1) - (sqrt(x)+1)/(x-1)", &mut ctx).expect("source");
        let target = cas_parser::parse("0", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_rationalized_target_aware(&mut ctx, source, target).expect("rewrite");
        assert_eq!(
            rewrite.kind,
            super::RationalizeRewriteKind::CancelToZeroAfterRationalize
        );
        assert_eq!(rewrite.rewritten, target);
    }
}
