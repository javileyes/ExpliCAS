use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_math::expr_predicates::contains_named_var;
use cas_math::expr_predicates::{is_one_expr, is_zero_expr};
use cas_math::expr_rewrite::smart_mul;
use cas_math::trig_roots_flatten::{flatten_add_sub_chain, flatten_mul_chain};
use std::cmp::Ordering;
use std::collections::BTreeMap;

use super::strong_target_match;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CombineLikeTermsRewrite {
    pub(crate) rewritten: ExprId,
    pub(crate) local_before: ExprId,
    pub(crate) local_after: ExprId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CollectTargetRewrite {
    pub(crate) focus_label: String,
    pub(crate) rewritten: ExprId,
}

pub(crate) fn try_rewrite_combine_like_terms_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<CombineLikeTermsRewrite> {
    if !matches!(ctx.get(source_expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    for target_term in flatten_add_sub_chain(ctx, target_expr) {
        let (_, focus_expr, _, _) = split_coeff_and_monomial(ctx, target_term);
        if is_one_expr(ctx, focus_expr) {
            continue;
        }

        let rewritten = rewrite_source_by_focus_simplifying_coeff(ctx, source_expr, focus_expr)?;
        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(CombineLikeTermsRewrite {
                rewritten,
                local_before: source_expr,
                local_after: rewritten,
            });
        }
    }

    try_rewrite_duplicate_addends_target_aware(ctx, source_expr, target_expr)
}

pub(crate) fn run_combine_like_terms_rewrite(
    ctx: &mut Context,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<crate::Step>)> {
    let rewrite = try_rewrite_combine_like_terms_target_aware(ctx, expr, target_expr)?;
    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            "Combine like terms",
            "Combine Like Terms",
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(ctx),
            rewrite.local_before,
            rewrite.local_after,
        );
        step.importance = crate::ImportanceLevel::Medium;
        step.category = crate::StepCategory::Simplify;
        vec![step]
    } else {
        Vec::new()
    };

    Some((rewrite.rewritten, steps))
}

pub(crate) fn try_rewrite_collect_monomial_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<CollectTargetRewrite> {
    for target_term in flatten_add_sub_chain(ctx, target_expr) {
        let (coeff_expr, focus_expr, _, focus_label) = split_coeff_and_monomial(ctx, target_term);
        if is_one_expr(ctx, focus_expr)
            || should_skip_simple_collect_focus(ctx, coeff_expr, &focus_label)
        {
            continue;
        }

        let rewritten = rewrite_source_by_focus(ctx, source_expr, focus_expr)?;
        if matches_target_modulo_simplify(ctx, rewritten, target_expr) {
            return Some(CollectTargetRewrite {
                focus_label,
                rewritten,
            });
        }
    }
    None
}

fn rewrite_source_by_focus(
    ctx: &mut Context,
    source_expr: ExprId,
    focus_expr: ExprId,
) -> Option<ExprId> {
    let focus_signature = monomial_signature(ctx, focus_expr)?;
    let source_terms = flatten_add_sub_chain(ctx, source_expr);
    if source_terms.len() < 2 {
        return None;
    }

    let mut grouped_coeffs = Vec::new();
    let mut rebuilt_terms = Vec::new();
    let mut first_group_index = None;

    for term in source_terms {
        if let Some(coeff) = divide_term_by_focus(ctx, term, &focus_signature) {
            if first_group_index.is_none() {
                first_group_index = Some(rebuilt_terms.len());
            }
            grouped_coeffs.push(coeff);
        } else {
            rebuilt_terms.push(term);
        }
    }

    if grouped_coeffs.len() < 2 {
        return None;
    }

    let grouped_coeff = combine_add_chain(ctx, &grouped_coeffs);
    let grouped_term = if is_zero_expr(ctx, grouped_coeff) {
        return None;
    } else if is_one_expr(ctx, grouped_coeff) {
        focus_expr
    } else {
        smart_mul(ctx, grouped_coeff, focus_expr)
    };

    rebuilt_terms.insert(first_group_index.unwrap_or(0), grouped_term);
    Some(combine_add_chain(ctx, &rebuilt_terms))
}

fn rewrite_source_by_focus_simplifying_coeff(
    ctx: &mut Context,
    source_expr: ExprId,
    focus_expr: ExprId,
) -> Option<ExprId> {
    let focus_signature = monomial_signature(ctx, focus_expr)?;
    let source_terms = flatten_add_sub_chain(ctx, source_expr);
    if source_terms.len() < 2 {
        return None;
    }

    let mut grouped_coeffs = Vec::new();
    let mut rebuilt_terms = Vec::new();
    let mut first_group_index = None;

    for term in source_terms {
        if let Some(coeff) = divide_term_by_focus(ctx, term, &focus_signature) {
            if first_group_index.is_none() {
                first_group_index = Some(rebuilt_terms.len());
            }
            grouped_coeffs.push(coeff);
        } else {
            rebuilt_terms.push(term);
        }
    }

    if grouped_coeffs.len() < 2 {
        return None;
    }

    let grouped_coeff_sum = combine_add_chain(ctx, &grouped_coeffs);
    let grouped_coeff = run_default_simplify(ctx, grouped_coeff_sum);
    if !is_zero_expr(ctx, grouped_coeff) {
        let grouped_term = if is_one_expr(ctx, grouped_coeff) {
            focus_expr
        } else {
            smart_mul(ctx, grouped_coeff, focus_expr)
        };
        rebuilt_terms.insert(first_group_index.unwrap_or(0), grouped_term);
    }

    let rebuilt_sum = combine_add_chain(ctx, &rebuilt_terms);
    let rebuilt = run_default_simplify(ctx, rebuilt_sum);
    (compare_expr(ctx, rebuilt, source_expr) != Ordering::Equal).then_some(rebuilt)
}

fn combine_add_chain(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    match terms {
        [] => ctx.num(0),
        [only] => *only,
        [first, rest @ ..] => {
            let mut result = *first;
            for term in rest {
                result = ctx.add(Expr::Add(result, *term));
            }
            result
        }
    }
}

fn try_rewrite_duplicate_addends_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<CombineLikeTermsRewrite> {
    let Expr::Add(left, right) = ctx.get(source_expr) else {
        return None;
    };
    let left = *left;
    let right = *right;
    if compare_expr(ctx, left, right) != Ordering::Equal {
        return None;
    }

    let two = ctx.num(2);
    let doubled = smart_mul(ctx, two, left);
    let rewritten = run_default_simplify(ctx, doubled);
    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(CombineLikeTermsRewrite {
        rewritten,
        local_before: source_expr,
        local_after: rewritten,
    })
}

fn split_coeff_and_monomial(ctx: &mut Context, term: ExprId) -> (ExprId, ExprId, String, String) {
    let factors = flatten_mul_chain(ctx, term);
    let mut literal_factors = Vec::new();
    let mut coeff_factors = Vec::new();

    for factor in factors {
        if is_monomial_factor(ctx, factor) {
            literal_factors.push((render_expr(ctx, factor), factor));
        } else {
            coeff_factors.push(factor);
        }
    }

    literal_factors.sort_by(|left, right| left.0.cmp(&right.0));

    let literal = if literal_factors.is_empty() {
        ctx.num(1)
    } else {
        let mut result = literal_factors[0].1;
        for (_, factor) in literal_factors.iter().skip(1) {
            result = smart_mul(ctx, result, *factor);
        }
        result
    };
    let coeff = if coeff_factors.is_empty() {
        ctx.num(1)
    } else {
        let mut result = coeff_factors[0];
        for factor in coeff_factors.into_iter().skip(1) {
            result = smart_mul(ctx, result, factor);
        }
        result
    };

    let literal_label = render_expr(ctx, literal);
    (coeff, literal, literal_label.clone(), literal_label)
}

fn divide_term_by_focus(
    ctx: &mut Context,
    term: ExprId,
    focus_signature: &BTreeMap<String, i64>,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, term);
    let mut literal_counts = BTreeMap::new();
    let mut coeff_factors = Vec::new();

    for factor in factors {
        if let Some((var_name, degree)) = variable_power(ctx, factor) {
            *literal_counts.entry(var_name).or_insert(0) += degree;
        } else {
            coeff_factors.push(factor);
        }
    }

    for (var_name, needed_degree) in focus_signature {
        let available = literal_counts.get_mut(var_name)?;
        if *available < *needed_degree {
            return None;
        }
        *available -= *needed_degree;
    }

    for (var_name, remaining_degree) in literal_counts {
        if remaining_degree == 0 {
            continue;
        }
        let var_expr = ctx.var(&var_name);
        let factor = if remaining_degree == 1 {
            var_expr
        } else {
            let degree_expr = ctx.num(remaining_degree);
            ctx.add(Expr::Pow(var_expr, degree_expr))
        };
        coeff_factors.push(factor);
    }

    Some(combine_mul_chain(ctx, &coeff_factors))
}

fn combine_mul_chain(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    match factors {
        [] => ctx.num(1),
        [only] => *only,
        [first, rest @ ..] => {
            let mut result = *first;
            for factor in rest {
                result = smart_mul(ctx, result, *factor);
            }
            result
        }
    }
}

fn monomial_signature(ctx: &mut Context, expr: ExprId) -> Option<BTreeMap<String, i64>> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut signature = BTreeMap::new();

    for factor in factors {
        let (var_name, degree) = variable_power(ctx, factor)?;
        *signature.entry(var_name).or_insert(0) += degree;
    }

    (!signature.is_empty()).then_some(signature)
}

fn is_monomial_factor(ctx: &Context, expr: ExprId) -> bool {
    variable_power(ctx, expr).is_some()
}

fn variable_power(ctx: &Context, expr: ExprId) -> Option<(String, i64)> {
    match ctx.get(expr) {
        Expr::Variable(sym_id) => Some((ctx.sym_name(*sym_id).to_string(), 1)),
        Expr::Pow(base, exp) => match (ctx.get(*base), ctx.get(*exp)) {
            (Expr::Variable(sym_id), Expr::Number(n))
                if n.is_integer() && n.to_integer() >= 1.into() =>
            {
                Some((
                    ctx.sym_name(*sym_id).to_string(),
                    n.to_integer().try_into().ok()?,
                ))
            }
            _ => None,
        },
        _ => None,
    }
}

fn render_expr(ctx: &Context, expr: ExprId) -> String {
    format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: expr
        }
    )
}

fn matches_target_modulo_simplify(ctx: &mut Context, left: ExprId, right: ExprId) -> bool {
    if strong_target_match(ctx, left, right) {
        return true;
    }

    let zero = ctx.num(0);
    let difference = ctx.add(Expr::Sub(left, right));
    let simplified = run_default_simplify(ctx, difference);
    strong_target_match(ctx, simplified, zero)
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

fn is_simple_collect_focus(focus: &str) -> bool {
    !focus.is_empty()
        && focus
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn should_skip_simple_collect_focus(ctx: &Context, coeff_expr: ExprId, focus: &str) -> bool {
    is_simple_collect_focus(focus) && !contains_named_var(ctx, coeff_expr, focus)
}

#[cfg(test)]
mod tests {
    use super::{
        render_expr, run_combine_like_terms_rewrite, try_rewrite_collect_monomial_target_aware,
        try_rewrite_combine_like_terms_target_aware,
    };

    #[test]
    fn target_aware_combine_like_terms_matches_tabulated_targets() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*x + 3*x + 0", &mut ctx).expect("parse source");
        let target = cas_parser::parse("5*x", &mut ctx).expect("parse target");

        let rewrite = try_rewrite_combine_like_terms_target_aware(&mut ctx, source, target)
            .expect("combine-like-terms rewrite");
        assert_eq!(render_expr(&ctx, rewrite.rewritten), "5 * x");

        let (_rewritten, steps) =
            run_combine_like_terms_rewrite(&mut ctx, source, target, true).expect("stage rewrite");
        assert!(!steps.is_empty());
        assert!(steps
            .iter()
            .all(|step| step.rule_name == "Combine Like Terms"));
    }

    #[test]
    fn target_aware_combine_like_terms_matches_duplicate_addends() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("x + x", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*x", &mut ctx).expect("parse target");

        let rewrite = try_rewrite_combine_like_terms_target_aware(&mut ctx, source, target)
            .expect("duplicate-addends combine-like-terms rewrite");
        assert_eq!(render_expr(&ctx, rewrite.rewritten), "2 * x");
    }

    #[test]
    fn target_aware_collect_matches_tabulated_composite_monomial_targets() {
        let cases = [
            ("a*x*y + b*x*y + c", "(a+b)*x*y + c", &["x", "y"][..]),
            (
                "a*x^2*y + b*x^2*y + c",
                "(a+b)*x^2*y + c",
                &["x^2", "y"][..],
            ),
            ("a*y*z + b*y*z + c", "(a+b)*y*z + c", &["y", "z"][..]),
            (
                "a*x*y + b*x*y + c*x*z + d*x*z + e",
                "(a+b)*x*y + (c+d)*x*z + e",
                &["x", "y"][..],
            ),
        ];

        for (source, target, focus_fragments) in cases {
            let mut ctx = cas_ast::Context::new();
            let source = cas_parser::parse(source, &mut ctx).expect("parse source");
            let target = cas_parser::parse(target, &mut ctx).expect("parse target");
            let rewrite = try_rewrite_collect_monomial_target_aware(&mut ctx, source, target)
                .expect("expected target-aware collect rewrite");
            for fragment in focus_fragments {
                assert!(
                    rewrite.focus_label.contains(fragment),
                    "expected focus label `{}` to contain `{fragment}`",
                    rewrite.focus_label
                );
            }
        }
    }

    #[test]
    fn ignores_simple_variable_collect_targets() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("a*x + b*x + c", &mut ctx).expect("parse source");
        let target = cas_parser::parse("(a+b)*x + c", &mut ctx).expect("parse target");
        assert!(try_rewrite_collect_monomial_target_aware(&mut ctx, source, target).is_none());
    }

    #[test]
    fn allows_simple_focus_when_target_coefficient_depends_on_same_variable() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("x^2 + 2*x + 1", &mut ctx).expect("parse source");
        let target = cas_parser::parse("x*(x + 2) + 1", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_collect_monomial_target_aware(&mut ctx, source, target)
            .expect("expected target-aware collect rewrite");

        assert_eq!(rewrite.focus_label, "x");
    }
}
