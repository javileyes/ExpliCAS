use super::strong_target_match;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_nary::{add_terms_signed, mul_leaves, Sign};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RadicalRewriteKind {
    SqrtPerfectSquare,
    SquareOfSquareRoot,
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
            Self::SquareOfSquareRoot => "Square a radical under its domain condition",
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::SqrtPerfectSquare => "Sqrt Perfect Square",
            Self::SquareOfSquareRoot => "Square of Square Root",
        }
    }
}

pub(crate) fn try_rewrite_radical_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<RadicalRewrite> {
    if let Some(rewrite) = try_rewrite_direct_radical_target_aware(ctx, source_expr, target_expr) {
        return Some(rewrite);
    }

    try_rewrite_additive_passthrough_radical_target_aware(ctx, source_expr, target_expr)
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

pub(crate) fn try_rewrite_odd_half_power_to_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExprId> {
    if let Some(rewritten) = try_rewrite_odd_half_power_with_optional_simplify(ctx, source_expr) {
        if odd_half_power_target_match(ctx, rewritten, target_expr) {
            return Some(target_expr);
        }
    }

    let source_terms = signed_additive_terms(ctx, source_expr);
    let target_terms = signed_additive_terms(ctx, target_expr);
    if source_terms.len() < 2 || target_terms.len() != source_terms.len() {
        return None;
    }

    for (source_index, source_focus) in source_terms.iter().copied().enumerate() {
        let Some(rewritten) = try_rewrite_odd_half_power_with_optional_simplify(ctx, source_focus)
        else {
            continue;
        };

        for (target_index, target_focus) in target_terms.iter().copied().enumerate() {
            if !odd_half_power_target_match(ctx, rewritten, target_focus) {
                continue;
            }

            let source_passthrough =
                collect_passthrough_terms_excluding_index(&source_terms, source_index);
            let target_passthrough =
                collect_passthrough_terms_excluding_index(&target_terms, target_index);
            if additive_term_multiset_matches(ctx, &source_passthrough, &target_passthrough) {
                return Some(target_expr);
            }
        }
    }

    None
}

fn odd_half_power_target_match(ctx: &mut Context, rewritten: ExprId, target_expr: ExprId) -> bool {
    if strong_target_match(ctx, rewritten, target_expr) {
        return true;
    }

    if odd_half_power_domain_equivalent_target_match(ctx, rewritten, target_expr) {
        return true;
    }

    let simplified = run_default_simplify(ctx, rewritten);
    simplified != rewritten
        && (strong_target_match(ctx, simplified, target_expr)
            || odd_half_power_domain_equivalent_target_match(ctx, simplified, target_expr))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OddHalfPowerProductForm {
    base: ExprId,
    outside_power: i64,
}

fn odd_half_power_domain_equivalent_target_match(
    ctx: &mut Context,
    rewritten: ExprId,
    target_expr: ExprId,
) -> bool {
    let Some(rewritten_form) = extract_odd_half_power_product_form(ctx, rewritten) else {
        return false;
    };
    let Some(target_form) = extract_odd_half_power_product_form(ctx, target_expr) else {
        return false;
    };

    rewritten_form.outside_power == target_form.outside_power
        && compare_expr(ctx, rewritten_form.base, target_form.base) == Ordering::Equal
}

fn extract_odd_half_power_product_form(
    ctx: &Context,
    expr: ExprId,
) -> Option<OddHalfPowerProductForm> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (sqrt_index, sqrt_factor) in factors.iter().copied().enumerate() {
        let Some(base) = extract_sqrt_argument(ctx, sqrt_factor) else {
            continue;
        };
        let outer_factor = factors[1 - sqrt_index];
        let Some((outer_base, outside_power)) =
            extract_odd_half_power_outer_factor(ctx, outer_factor)
        else {
            continue;
        };
        if compare_expr(ctx, outer_base, base) == Ordering::Equal {
            return Some(OddHalfPowerProductForm {
                base,
                outside_power,
            });
        }
    }

    None
}

fn extract_odd_half_power_outer_factor(ctx: &Context, expr: ExprId) -> Option<(ExprId, i64)> {
    if let Some(inner) = abs_argument(ctx, expr) {
        return Some((inner, 1));
    }

    match ctx.get(expr) {
        Expr::Pow(base, exponent) => {
            let power = small_positive_integer_value(ctx, *exponent)?;
            if let Some(inner) = abs_argument(ctx, *base) {
                Some((inner, power))
            } else {
                Some((*base, power))
            }
        }
        _ => Some((expr, 1)),
    }
}

fn abs_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn small_positive_integer_value(ctx: &Context, expr: ExprId) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Number(n)
            if n.is_integer() && *n > num_rational::BigRational::from_integer(0.into()) =>
        {
            n.to_integer().try_into().ok()
        }
        _ => None,
    }
}

fn try_rewrite_odd_half_power_with_optional_simplify(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if let Some(rewritten) = try_rewrite_odd_half_power_target_aware(ctx, expr) {
        return Some(rewritten);
    }

    let simplified = run_default_simplify(ctx, expr);
    if simplified == expr {
        return None;
    }

    try_rewrite_odd_half_power_target_aware(ctx, simplified)
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

fn try_rewrite_direct_radical_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<RadicalRewrite> {
    if let Some(rewrite) =
        try_rewrite_square_of_square_root_target_aware(ctx, source_expr, target_expr)
    {
        return Some(rewrite);
    }

    let rewrite =
        cas_math::perfect_square_support::try_rewrite_sqrt_perfect_square_expr(ctx, source_expr)?;
    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    let radicand = extract_sqrt_argument(ctx, source_expr)?;
    Some(RadicalRewrite {
        rewritten: target_expr,
        kind: RadicalRewriteKind::SqrtPerfectSquare,
        required_conditions: vec![crate::ImplicitCondition::NonNegative(radicand)],
    })
}

fn try_rewrite_square_of_square_root_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<RadicalRewrite> {
    let (base, exponent) = match ctx.get(source_expr) {
        Expr::Pow(base, exponent) => (*base, *exponent),
        _ => return None,
    };
    if small_positive_integer_value(ctx, exponent)? != 2 {
        return None;
    }

    let radicand = extract_sqrt_argument(ctx, base)?;
    if !strong_target_match(ctx, radicand, target_expr) {
        return None;
    }

    Some(RadicalRewrite {
        rewritten: target_expr,
        kind: RadicalRewriteKind::SquareOfSquareRoot,
        required_conditions: vec![crate::ImplicitCondition::NonNegative(radicand)],
    })
}

fn try_rewrite_additive_passthrough_radical_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<RadicalRewrite> {
    let source_terms = signed_additive_terms(ctx, source_expr);
    let target_terms = signed_additive_terms(ctx, target_expr);
    if source_terms.len() < 2 || target_terms.len() != source_terms.len() {
        return None;
    }

    for (source_index, source_focus) in source_terms.iter().copied().enumerate() {
        for (target_index, target_focus) in target_terms.iter().copied().enumerate() {
            let Some(rewrite) =
                try_rewrite_direct_radical_target_aware(ctx, source_focus, target_focus)
            else {
                continue;
            };

            let source_passthrough =
                collect_passthrough_terms_excluding_index(&source_terms, source_index);
            let target_passthrough =
                collect_passthrough_terms_excluding_index(&target_terms, target_index);
            if additive_term_multiset_matches(ctx, &source_passthrough, &target_passthrough) {
                return Some(RadicalRewrite {
                    rewritten: target_expr,
                    kind: rewrite.kind,
                    required_conditions: rewrite.required_conditions,
                });
            }
        }
    }

    None
}

fn signed_additive_terms(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    add_terms_signed(ctx, expr)
        .into_iter()
        .map(|(term, sign)| apply_sign_to_term(ctx, term, sign))
        .collect()
}

fn apply_sign_to_term(ctx: &mut Context, term: ExprId, sign: Sign) -> ExprId {
    match sign {
        Sign::Pos => term,
        Sign::Neg => ctx.add(Expr::Neg(term)),
    }
}

fn collect_passthrough_terms_excluding_index(
    terms: &[ExprId],
    excluded_index: usize,
) -> Vec<ExprId> {
    terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| (index != excluded_index).then_some(*term))
        .collect()
}

fn additive_term_multiset_matches(
    ctx: &mut Context,
    lhs_terms: &[ExprId],
    rhs_terms: &[ExprId],
) -> bool {
    if lhs_terms.len() != rhs_terms.len() {
        return false;
    }

    let mut lhs = lhs_terms.to_vec();
    let mut rhs = rhs_terms.to_vec();
    lhs.sort_by(|left, right| compare_expr(ctx, *left, *right));
    rhs.sort_by(|left, right| compare_expr(ctx, *left, *right));

    lhs.iter()
        .zip(rhs.iter())
        .all(|(left, right)| compare_expr(ctx, *left, *right) == Ordering::Equal)
}

fn run_default_simplify(ctx: &mut Context, expr: ExprId) -> ExprId {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (rewritten, _steps, _stats) = simplifier.simplify_with_stats(
        expr,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..crate::SimplifyOptions::default()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);
    rewritten
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_odd_half_power_target_aware, try_rewrite_odd_half_power_to_target_aware,
        try_rewrite_radical_target_aware,
    };
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
    fn rewrites_odd_half_power_with_additive_passthrough_target_aware() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x^3)+a", &mut ctx).expect("expr");
        let target = parse("abs(x)*sqrt(x)+a", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_odd_half_power_to_target_aware(&mut ctx, expr, target).expect("rewrite");
        assert_eq!(rewrite, target);
    }

    #[test]
    fn rewrites_higher_odd_half_power_to_nonnegative_target_aware() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x^5)", &mut ctx).expect("expr");
        let target = parse("x^2*sqrt(x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_odd_half_power_to_target_aware(&mut ctx, expr, target).expect("rewrite");
        assert_eq!(rewrite, target);
    }

    #[test]
    fn rewrites_higher_odd_half_power_with_additive_passthrough_to_nonnegative_target_aware() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x^7)+a", &mut ctx).expect("expr");
        let target = parse("x^3*sqrt(x)+a", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_odd_half_power_to_target_aware(&mut ctx, expr, target).expect("rewrite");
        assert_eq!(rewrite, target);
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

    #[test]
    fn rewrites_sqrt_squared_symbol_target_aware() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x^2)", &mut ctx).expect("expr");
        let target = parse("abs(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_radical_target_aware(&mut ctx, expr, target).expect("rewrite");
        let text = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(text.contains("|x|") || text.contains("abs(x)"));
    }

    #[test]
    fn rewrites_square_of_square_root_target_aware() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x)^2", &mut ctx).expect("expr");
        let target = parse("x", &mut ctx).expect("target");
        let rewrite = try_rewrite_radical_target_aware(&mut ctx, expr, target).expect("rewrite");
        let text = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(text, "x");
        assert_eq!(rewrite.required_conditions.len(), 1);
    }

    #[test]
    fn rewrites_sqrt_perfect_square_with_additive_passthrough_target_aware() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(a^2 + 2*a*b + b^2)+c", &mut ctx).expect("expr");
        let target = parse("abs(a+b)+c", &mut ctx).expect("target");
        let rewrite = try_rewrite_radical_target_aware(&mut ctx, expr, target).expect("rewrite");
        let text = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(text.contains("|a + b| + c") || text.contains("abs(a + b) + c"));
    }
}
