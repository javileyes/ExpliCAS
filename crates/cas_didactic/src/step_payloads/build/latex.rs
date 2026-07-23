mod rule;

use crate::runtime::Step;
use cas_ast::Context;
use cas_formatter::LaTeXExpr;
use cas_solver_core::rule_names::{
    RULE_CONSERVAR_DERIVADA_RESIDUAL, RULE_CONSERVAR_INTEGRAL_RESIDUAL,
    RULE_CONSERVAR_LIMITE_RESIDUAL,
};

pub(super) struct RenderedStepWireLatex {
    pub(super) before_latex: String,
    pub(super) after_latex: String,
    pub(super) rule_latex: String,
}

pub(super) fn render_step_wire_latex(
    context: &Context,
    step: &Step,
    is_first: bool,
) -> RenderedStepWireLatex {
    let (mut before_latex, mut after_latex) = if matches!(
        step.rule_name.as_str(),
        "Product-to-Sum Identity" | "Hyperbolic Product-to-Sum Identity"
    ) || matches!(
        step.rule_name.as_str(),
        RULE_CONSERVAR_DERIVADA_RESIDUAL
            | RULE_CONSERVAR_INTEGRAL_RESIDUAL
            | RULE_CONSERVAR_LIMITE_RESIDUAL
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
    let rule_latex = derive_rule_latex_from_global(step, &before_latex, &after_latex, is_first)
        .unwrap_or_else(|| render_normalized_rule_latex(context, step, is_first));
    RenderedStepWireLatex {
        before_latex,
        after_latex,
        rule_latex,
    }
}

/// Extract the single brace-balanced `\color{<color>}{…}` span of a rendered
/// global state. Returns `(span, is_full)`; `None` when the state has zero or
/// several spans of that color (multi-term highlights keep the local render).
fn single_color_span(latex: &str, color: &str) -> Option<(String, bool)> {
    let needle = format!("\\color{{{}}}{{", color);
    let start = latex.find(&needle)?;
    let bytes = latex.as_bytes();
    let span_start = start + needle.len();
    let mut depth = 1usize;
    let mut k = span_start;
    while k < bytes.len() && depth > 0 {
        match bytes[k] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            _ => {}
        }
        k += 1;
    }
    if depth != 0 || latex[k..].contains(&needle) {
        return None;
    }
    let span = latex[span_start..k - 1].to_string();
    // Wrapper shape `{\color{c}{SPAN}}`: after the span's closing brace only
    // the wrapper's final `}` may remain for the span to cover the full state.
    let full = start == 1 && latex.starts_with('{') && latex[k..].trim_end() == "}";
    Some((span, full))
}

/// Coherencia local-vs-global (auditoría educativa 2026-07-23, ciclo E2): el
/// fragmento resaltado del `rule_latex` se EXTRAE del mismo render global que
/// el usuario ve en `before_latex`/`after_latex`, así el zoom siempre muestra
/// una forma que existe en la cadena visible (misma tabla de nombres, mismo
/// orden de términos). Un rojo a-pantalla-completa solo se acepta en los
/// verbos de cálculo y en el PRIMER paso del wire (ahí el "sitio" ES la
/// expresión entera: d/dx, ∫, Σ, taylor, apart, verbos de matriz…); en el
/// resto se conserva el render local, que al menos enseña el sitio real de la
/// regla.
fn derive_rule_latex_from_global(
    step: &Step,
    before_latex: &str,
    after_latex: &str,
    is_first: bool,
) -> Option<String> {
    let (red, red_full) = single_color_span(before_latex, "red")?;
    let (green, _) = single_color_span(after_latex, "green")?;
    let full_red_ok = is_first
        || matches!(
            step.rule_name.as_str(),
            "Symbolic Differentiation" | "Symbolic Integration" | "Higher-Order Differentiation"
        )
        || step.rule_name.starts_with("Evaluar límite")
        || step.rule_name == "Calcular el límite";
    if red_full && !full_red_ok {
        return None;
    }
    Some(format!(
        "{{\\color{{red}}{{{}}}}} \\rightarrow {{\\color{{green}}{{{}}}}}",
        red, green
    ))
}

/// Coherent-highlight policy (auditoría educativa 2026-07-23): the GREEN side
/// always shows the display-normalized result (never the raw rewrite output
/// like `y·2·x^{2-1}`), and the RED side normalizes too except on the first
/// step (which keeps the user's input form). This generalizes the per-rule
/// display patches above: the highlight must show what the visible chain
/// shows.
fn render_normalized_rule_latex(context: &Context, step: &Step, is_first: bool) -> String {
    let focus_before = step.before_local().unwrap_or(step.before);
    let focus_after = step.after_local().unwrap_or(step.after);
    let mut temp_ctx = context.clone();
    let folded_after =
        super::expr::cleanup_symbolic_diff_after_for_display(&mut temp_ctx, focus_after);
    let norm_after = cas_solver_core::eval_step_pipeline::normalize_expr_for_display(
        &mut temp_ctx,
        folded_after,
    );
    let norm_before = if is_first {
        focus_before
    } else {
        let folded =
            super::expr::cleanup_symbolic_diff_after_for_display(&mut temp_ctx, focus_before);
        cas_solver_core::eval_step_pipeline::normalize_expr_for_display(&mut temp_ctx, folded)
    };
    crate::step_payload_render::render_local_rule_latex(&temp_ctx, norm_before, norm_after)
}
