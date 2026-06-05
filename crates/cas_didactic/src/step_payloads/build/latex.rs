mod rule;

use crate::runtime::Step;
use cas_ast::Context;
use cas_formatter::LaTeXExpr;

pub(super) struct RenderedStepWireLatex {
    pub(super) before_latex: String,
    pub(super) after_latex: String,
    pub(super) rule_latex: String,
}

pub(super) fn render_step_wire_latex(context: &Context, step: &Step) -> RenderedStepWireLatex {
    let (mut before_latex, mut after_latex) = if matches!(
        step.rule_name.as_str(),
        "Product-to-Sum Identity" | "Hyperbolic Product-to-Sum Identity"
    ) || matches!(
        step.rule_name.as_str(),
        "Conservar derivada residual" | "Conservar integral residual" | "Conservar límite residual"
    ) {
        let before_expr = step.global_before.unwrap_or(step.before);
        let after_expr = step.global_after.unwrap_or(step.after);
        (
            LaTeXExpr {
                context,
                id: before_expr,
            }
            .to_latex(),
            LaTeXExpr {
                context,
                id: after_expr,
            }
            .to_latex(),
        )
    } else {
        crate::timeline::simplify_highlights::render_step_wire_global_before_after_latex(
            context, step,
        )
    };
    if step.rule_name == "Symbolic Differentiation" {
        let mut temp_ctx = context.clone();
        let snapshots = crate::timeline::simplify_highlights::step_wire_presentation_snapshots(
            &mut temp_ctx,
            step,
        );
        let raw_after = snapshots.global_after_expr;
        if super::expr::contains_ln_e_call(&temp_ctx, raw_after)
            || super::expr::contains_positive_integer_power_exponent_arithmetic(
                &temp_ctx, raw_after,
            )
            || super::expr::contains_nested_reciprocal_division(&temp_ctx, raw_after)
        {
            let collapsed =
                super::expr::cleanup_symbolic_diff_after_for_display(&mut temp_ctx, raw_after);
            let normalized = cas_solver_core::eval_step_pipeline::normalize_expr_for_display(
                &mut temp_ctx,
                collapsed,
            );
            after_latex = format!(
                "{{\\color{{green}}{{{}}}}}",
                LaTeXExpr {
                    context: &temp_ctx,
                    id: normalized
                }
                .to_latex()
            );
        }
    }
    let mut temp_ctx = context.clone();
    let snapshots =
        crate::timeline::simplify_highlights::step_wire_presentation_snapshots(&mut temp_ctx, step);
    let raw_before = snapshots.global_before_expr;
    if step.rule_name == "Symbolic Integration"
        && super::expr::contains_negative_half_power(&temp_ctx, raw_before)
    {
        let collapsed =
            super::expr::collapse_negative_half_powers_for_display(&mut temp_ctx, raw_before);
        let normalized = cas_solver_core::eval_step_pipeline::normalize_expr_for_display(
            &mut temp_ctx,
            collapsed,
        );
        before_latex = format!(
            "{{\\color{{red}}{{{}}}}}",
            LaTeXExpr {
                context: &temp_ctx,
                id: normalized
            }
            .to_latex()
        );
    }
    let mut temp_ctx = context.clone();
    let snapshots =
        crate::timeline::simplify_highlights::step_wire_presentation_snapshots(&mut temp_ctx, step);
    let raw_before = snapshots.global_before_expr;
    if super::expr::cleanup_step_prefers_compact_nested_reciprocal_display(
        &temp_ctx, step, raw_before,
    ) {
        let collapsed =
            super::expr::cleanup_symbolic_diff_after_for_display(&mut temp_ctx, raw_before);
        let normalized = cas_solver_core::eval_step_pipeline::normalize_expr_for_display(
            &mut temp_ctx,
            collapsed,
        );
        before_latex = format!(
            "{{\\color{{red}}{{{}}}}}",
            LaTeXExpr {
                context: &temp_ctx,
                id: normalized
            }
            .to_latex()
        );
    }
    let mut temp_ctx = context.clone();
    let snapshots =
        crate::timeline::simplify_highlights::step_wire_presentation_snapshots(&mut temp_ctx, step);
    let raw_after = snapshots.global_after_expr;
    if super::expr::cleanup_step_prefers_compact_nested_reciprocal_display(
        &temp_ctx, step, raw_after,
    ) {
        let collapsed =
            super::expr::cleanup_symbolic_diff_after_for_display(&mut temp_ctx, raw_after);
        let normalized = cas_solver_core::eval_step_pipeline::normalize_expr_for_display(
            &mut temp_ctx,
            collapsed,
        );
        after_latex = format!(
            "{{\\color{{green}}{{{}}}}}",
            LaTeXExpr {
                context: &temp_ctx,
                id: normalized
            }
            .to_latex()
        );
    }
    RenderedStepWireLatex {
        before_latex,
        after_latex,
        rule_latex: rule::render_step_rule_latex(context, step),
    }
}
