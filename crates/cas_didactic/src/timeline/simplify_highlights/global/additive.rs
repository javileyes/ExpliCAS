mod focus;

use super::super::TimelineStepSnapshots;
use super::additive_render::{
    render_after_additive_focus, render_before_additive_focus,
    render_before_additive_focus_with_exact_paths,
};
use crate::runtime::Step;
use cas_ast::{Context, Expr, ExprPath};
use cas_formatter::path::extract_add_terms;
use cas_formatter::{DisplayContext, StylePreferences};
use cas_math::expr_nary::{add_terms_signed, Sign};
use num_traits::Zero;

pub(super) fn render_additive_focus_transition(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    focus_before: cas_ast::ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    let focus_after = step.after_local().unwrap_or(step.after);
    if matches!(context.get(focus_after), Expr::Number(n) if n.is_zero())
        && !matches!(context.get(snapshots.global_after_expr), Expr::Number(n) if n.is_zero())
    {
        let removed_paths = collect_removed_additive_term_paths(
            context,
            snapshots.global_before_expr,
            snapshots.global_after_expr,
        );
        let expected_removed_terms = add_terms_signed(context, focus_before).len();
        if !removed_paths.is_empty() && removed_paths.len() == expected_removed_terms {
            let before = render_before_additive_focus_with_exact_paths(
                context,
                snapshots.global_before_expr,
                &removed_paths,
                display_hints,
                style_prefs,
            );
            let after = render_after_additive_focus(
                context,
                snapshots.global_after_expr,
                focus_after,
                &removed_paths,
                display_hints,
                style_prefs,
            );
            return (before, after);
        }
    }

    let focus_terms = extract_add_terms(context, focus_before);
    let found_paths = focus::collect_additive_focus_paths_with_scope(
        context,
        step,
        snapshots.global_before_expr,
        focus_before,
        &focus_terms,
    );
    let before = render_before_additive_focus(
        context,
        snapshots.global_before_expr,
        focus_before,
        &found_paths,
        step,
        display_hints,
        style_prefs,
    );
    let after = render_after_additive_focus(
        context,
        snapshots.global_after_expr,
        focus_after,
        &found_paths,
        display_hints,
        style_prefs,
    );

    (before, after)
}

fn collect_removed_additive_term_paths(
    context: &Context,
    global_before_expr: cas_ast::ExprId,
    global_after_expr: cas_ast::ExprId,
) -> Vec<ExprPath> {
    let before_entries = collect_signed_additive_term_entries(context, global_before_expr);
    if before_entries.is_empty() {
        return Vec::new();
    }

    let after_terms = add_terms_signed(context, global_after_expr);
    let mut matched = vec![false; before_entries.len()];
    for (after_term, after_sign) in after_terms {
        if let Some((idx, _)) = before_entries.iter().enumerate().find(|(idx, entry)| {
            !matched[*idx]
                && entry.sign == after_sign
                && cas_ast::ordering::compare_expr(context, entry.term, after_term)
                    == std::cmp::Ordering::Equal
        }) {
            matched[idx] = true;
        }
    }

    before_entries
        .into_iter()
        .enumerate()
        .filter_map(|(idx, entry)| (!matched[idx]).then_some(entry.path))
        .collect()
}

#[derive(Clone)]
struct SignedAdditiveTermEntry {
    path: ExprPath,
    term: cas_ast::ExprId,
    sign: Sign,
}

fn collect_signed_additive_term_entries(
    context: &Context,
    expr: cas_ast::ExprId,
) -> Vec<SignedAdditiveTermEntry> {
    let mut entries = Vec::new();
    let mut path = ExprPath::new();
    collect_signed_additive_term_entries_rec(context, expr, Sign::Pos, &mut path, &mut entries);
    entries
}

fn collect_signed_additive_term_entries_rec(
    context: &Context,
    expr: cas_ast::ExprId,
    sign: Sign,
    path: &mut ExprPath,
    entries: &mut Vec<SignedAdditiveTermEntry>,
) {
    match context.get(expr) {
        Expr::Add(left, right) => {
            path.push(0);
            collect_signed_additive_term_entries_rec(context, *left, sign, path, entries);
            path.pop();

            path.push(1);
            collect_signed_additive_term_entries_rec(context, *right, sign, path, entries);
            path.pop();
        }
        Expr::Sub(left, right) => {
            path.push(0);
            collect_signed_additive_term_entries_rec(context, *left, sign, path, entries);
            path.pop();

            path.push(1);
            collect_signed_additive_term_entries_rec(context, *right, sign.negate(), path, entries);
            path.pop();
        }
        Expr::Neg(inner) => {
            if matches!(context.get(*inner), Expr::Add(_, _) | Expr::Sub(_, _)) {
                path.push(0);
                collect_signed_additive_term_entries_rec(
                    context,
                    *inner,
                    sign.negate(),
                    path,
                    entries,
                );
                path.pop();
            } else {
                path.push(0);
                entries.push(SignedAdditiveTermEntry {
                    path: path.clone(),
                    term: *inner,
                    sign: sign.negate(),
                });
                path.pop();
            }
        }
        _ => entries.push(SignedAdditiveTermEntry {
            path: path.clone(),
            term: expr,
            sign,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::collect_removed_additive_term_paths;
    use cas_formatter::path::navigate_to_subexpr;

    #[test]
    fn removed_additive_term_paths_keep_negative_trig_term() {
        let mut ctx = cas_ast::Context::new();
        let before_raw = cas_parser::parse(
            "tan(x) + atanh((x^2 - 1)/(x^2 + 1)) + sqrt(sqrt(x^2 - y^2) + x) + 1/tan(x) - ln(x) - 2/sin(2*x) - (sqrt(x + y) + sqrt(x - y))/sqrt(2)",
            &mut ctx,
        )
        .expect("parse before");
        let after_raw = cas_parser::parse(
            "atanh((x^2 - 1)/(x^2 + 1)) + sqrt(sqrt(x^2 - y^2) + x) - ln(x) - (sqrt(x + y) + sqrt(x - y))/sqrt(2)",
            &mut ctx,
        )
        .expect("parse after");
        let before = cas_solver_core::eval_step_pipeline::normalize_expr_for_display(&mut ctx, before_raw);
        let after = cas_solver_core::eval_step_pipeline::normalize_expr_for_display(&mut ctx, after_raw);

        let paths = collect_removed_additive_term_paths(&ctx, before, after);
        let removed: Vec<String> = paths
            .iter()
            .map(|path| {
                cas_formatter::clean_display_string(&format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &ctx,
                        id: navigate_to_subexpr(&ctx, before, path)
                    }
                ))
            })
            .collect();

        assert!(
            removed.iter().any(|item| item == "tan(x)"),
            "missing tan(x) in removed terms: {removed:?}"
        );
        assert!(
            removed.iter().any(|item| item == "1 / tan(x)"),
            "missing 1/tan(x) in removed terms: {removed:?}"
        );
        assert!(
            removed.iter().any(|item| item == "2 / sin(2 * x)"),
            "missing 2/sin(2*x) in removed terms: {removed:?}"
        );
    }

    #[test]
    fn removed_additive_term_paths_keep_negative_log_term() {
        let mut ctx = cas_ast::Context::new();
        let before_raw = cas_parser::parse(
            "atanh((x^2 - 1)/(x^2 + 1)) + sqrt(sqrt(x^2 - y^2) + x) - ln(x) - (sqrt(x + y) + sqrt(x - y))/sqrt(2)",
            &mut ctx,
        )
        .expect("parse before");
        let after_raw = cas_parser::parse(
            "sqrt(sqrt(x^2 - y^2) + x) - (sqrt(x + y) + sqrt(x - y))/sqrt(2)",
            &mut ctx,
        )
        .expect("parse after");
        let before = cas_solver_core::eval_step_pipeline::normalize_expr_for_display(&mut ctx, before_raw);
        let after = cas_solver_core::eval_step_pipeline::normalize_expr_for_display(&mut ctx, after_raw);

        let paths = collect_removed_additive_term_paths(&ctx, before, after);
        let removed: Vec<String> = paths
            .iter()
            .map(|path| {
                cas_formatter::clean_display_string(&format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &ctx,
                        id: navigate_to_subexpr(&ctx, before, path)
                    }
                ))
            })
            .collect();

        assert!(
            removed
                .iter()
                .any(|item| item.starts_with("atanh(") || item.starts_with("arctanh(")),
            "missing atanh(...) in removed terms: {removed:?}"
        );
        assert!(
            removed.iter().any(|item| item == "ln(x)"),
            "missing ln(x) in removed terms: {removed:?}"
        );
    }

    #[test]
    fn exact_path_render_keeps_negative_terms_in_before_latex() {
        let mut ctx = cas_ast::Context::new();
        let before_raw = cas_parser::parse(
            "atanh((x^2 - 1)/(x^2 + 1)) + sqrt(sqrt(x^2 - y^2) + x) - ln(x) - (sqrt(x + y) + sqrt(x - y))/sqrt(2)",
            &mut ctx,
        )
        .expect("parse before");
        let after_raw = cas_parser::parse(
            "sqrt(sqrt(x^2 - y^2) + x) - (sqrt(x + y) + sqrt(x - y))/sqrt(2)",
            &mut ctx,
        )
        .expect("parse after");
        let before =
            cas_solver_core::eval_step_pipeline::normalize_expr_for_display(&mut ctx, before_raw);
        let after =
            cas_solver_core::eval_step_pipeline::normalize_expr_for_display(&mut ctx, after_raw);
        let paths = collect_removed_additive_term_paths(&ctx, before, after);

        let latex =
            crate::timeline::simplify_highlights::global::additive_render::render_before_additive_focus_with_exact_paths(
                &ctx,
                before,
                &paths,
                &cas_formatter::DisplayContext::default(),
                &cas_formatter::StylePreferences::default(),
            );

        assert!(
            latex.contains("\\color{red}{\\text{atanh}")
                && latex.contains("\\color{red}{\\ln(x)}"),
            "expected exact-path render to keep both atanh and ln highlighted, got: {latex}"
        );
    }

    #[test]
    fn runtime_step_snapshots_keep_negative_terms_for_partitioned_zero_chunks() {
        let expr = "((cos(x))^3 - (3*cos(x) + cos(3*x))/4) + (tan(x) + 1/tan(x) - 2/sin(2*x)) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (sqrt(x + sqrt(x^2 - y^2)) - (sqrt(x+y) + sqrt(x-y))/sqrt(2))";
        let mut engine = cas_solver::runtime::Engine::new();
        let parsed = cas_parser::parse(expr, &mut engine.simplifier.context).expect("parse");
        let output = engine
            .eval_stateless(
                cas_solver::runtime::EvalOptions::default(),
                cas_solver::runtime::EvalRequest {
                    raw_input: expr.to_string(),
                    parsed,
                    action: cas_solver::runtime::EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval");

        let mut temp_ctx = engine.simplifier.context.clone();
        let step2 = &output.steps.as_slice()[1];
        let snapshots2 =
            crate::timeline::simplify_highlights::step_wire_presentation_snapshots(&mut temp_ctx, step2);
        let removed2 =
            collect_removed_additive_term_paths(&temp_ctx, snapshots2.global_before_expr, snapshots2.global_after_expr);
        let removed2_display: Vec<String> = removed2
            .iter()
            .map(|path| {
                cas_formatter::clean_display_string(&format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &temp_ctx,
                        id: navigate_to_subexpr(&temp_ctx, snapshots2.global_before_expr, path)
                    }
                ))
            })
            .collect();
        let rendered2 =
            crate::timeline::simplify_highlights::global::additive_render::render_before_additive_focus_with_exact_paths(
                &temp_ctx,
                snapshots2.global_before_expr,
                &removed2,
                &cas_formatter::DisplayContext::default(),
                &cas_formatter::StylePreferences::default(),
            );
        assert!(
            rendered2.contains("\\color{red}{\\tan(x)}")
                && rendered2.contains("\\color{red}{\\frac{2}{\\sin(2\\cdot x)}}"),
            "expected runtime step 2 exact render to keep tan and 2/sin highlighted, removed={removed2_display:?}, got: {rendered2}"
        );

        let step3 = &output.steps.as_slice()[2];
        let snapshots3 =
            crate::timeline::simplify_highlights::step_wire_presentation_snapshots(&mut temp_ctx, step3);
        let removed3 =
            collect_removed_additive_term_paths(&temp_ctx, snapshots3.global_before_expr, snapshots3.global_after_expr);
        let rendered3 =
            crate::timeline::simplify_highlights::global::additive_render::render_before_additive_focus_with_exact_paths(
                &temp_ctx,
                snapshots3.global_before_expr,
                &removed3,
                &cas_formatter::DisplayContext::default(),
                &cas_formatter::StylePreferences::default(),
            );
        assert!(
            rendered3.contains("\\color{red}{\\text{atanh}")
                && rendered3.contains("\\color{red}{\\ln(x)}"),
            "expected runtime step 3 exact render to keep atanh and ln highlighted, got: {rendered3}"
        );
    }
}
