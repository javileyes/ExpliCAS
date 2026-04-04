use super::strong_target_match;
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RadicalRewriteKind {
    SqrtPerfectSquare,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RadicalRewrite {
    pub(crate) rewritten: ExprId,
    pub(crate) kind: RadicalRewriteKind,
    pub(crate) required_conditions: Vec<crate::ImplicitCondition>,
}

impl RadicalRewriteKind {
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::SqrtPerfectSquare => "Take the square root of a perfect square",
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::SqrtPerfectSquare => "Sqrt Perfect Square",
        }
    }
}

pub(crate) fn try_rewrite_radical_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<RadicalRewrite> {
    let rewrite =
        cas_math::perfect_square_support::try_rewrite_sqrt_perfect_square_expr(ctx, source_expr)?;
    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    let radicand = extract_sqrt_argument(ctx, source_expr)?;
    Some(RadicalRewrite {
        rewritten: rewrite.rewritten,
        kind: RadicalRewriteKind::SqrtPerfectSquare,
        required_conditions: vec![crate::ImplicitCondition::NonNegative(radicand)],
    })
}

pub(crate) fn try_rewrite_odd_half_power_target_aware(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if let Some(rewrite) = cas_math::root_forms::try_rewrite_odd_half_power_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }

    let normalized = cas_math::canonical_forms::normalize_core(ctx, expr);
    if normalized == expr {
        return None;
    }

    cas_math::root_forms::try_rewrite_odd_half_power_expr(ctx, normalized)
        .map(|rewrite| rewrite.rewritten)
}

fn extract_sqrt_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            let half = num_rational::BigRational::new(1.into(), 2.into());
            match ctx.get(*exp) {
                Expr::Number(n) if *n == half => Some(*base),
                _ => None,
            }
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{try_rewrite_odd_half_power_target_aware, try_rewrite_radical_target_aware};
    use cas_ast::{Context, Expr};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn rewrites_direct_odd_half_power() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let exp = ctx.rational(3, 2);
        let expr = ctx.add(Expr::Pow(x, exp));
        let rewritten = try_rewrite_odd_half_power_target_aware(&mut ctx, expr).expect("rewrite");
        let text = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewritten
            }
        );
        assert!(text.contains("sqrt"));
        assert!(text.contains("|x|") || text.contains("abs"));
    }

    #[test]
    fn rewrites_sqrt_perfect_square_target_aware() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(a^2 + 2*a*b + b^2)", &mut ctx).expect("expr");
        let target = parse("abs(a+b)", &mut ctx).expect("target");
        let rewrite = try_rewrite_radical_target_aware(&mut ctx, expr, target).expect("rewrite");
        let text = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(text.contains("|a + b|") || text.contains("abs(a + b)"));
    }
}
