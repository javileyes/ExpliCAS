use super::super::payload::EvalOutputResultPayload;
use crate::eval_output_finalize::{EvalOutputWire, EvalOutputWireBuild};
use crate::eval_output_finalize_input::EvalOutputFinalizeShared;

/// Render a single required-display condition (its plain-text form) as LaTeX, matching the
/// `\begin{cases} … & \text{if } <cond> \end{cases}` form already used for in-set conditional
/// solution sets. Only the relational/operator glyphs differ between the two surfaces.
fn condition_to_latex(condition: &str) -> String {
    condition
        .replace('≥', "\\geq ")
        .replace('≤', "\\leq ")
        .replace('≠', "\\neq ")
        .replace('·', "\\cdot ")
}

/// A bare `All real numbers` solve result that carries domain `required_display` conditions is
/// dishonest in the default text surface: the true solution is `ℝ` RESTRICTED to those conditions
/// (e.g. `solve(ln(x²)=2·ln(x)) = ℝ ∩ {x>0}`). The conditions are computed and surfaced in the JSON
/// wire but dropped from `result`/`result_latex`. Inline them as `All real numbers if <conds>`,
/// matching the existing convention for an in-set `Conditional` result (`1/x=1/x → "… if x ≠ 0"`).
/// Scoped to the exact bare-`AllReals` string so it never double-renders an already-conditional set
/// nor touches the many `simplify` results whose `required_display` is intentionally JSON-only.
fn inline_all_reals_required_conditions(
    result: &str,
    required_display: &[String],
) -> Option<(String, String)> {
    if result != "All real numbers" || required_display.is_empty() {
        return None;
    }
    let joined = required_display.join(", ");
    let latex_conds = required_display
        .iter()
        .map(|c| condition_to_latex(c))
        .collect::<Vec<_>>()
        .join(", ");
    Some((
        format!("All real numbers if {joined}"),
        format!("\\begin{{cases}} \\mathbb{{R}} & \\text{{if }} {latex_conds} \\end{{cases}}"),
    ))
}

pub(super) fn build_eval_output(
    payload: EvalOutputResultPayload,
    steps_count: usize,
    shared: EvalOutputFinalizeShared<'_>,
    wire: Option<serde_json::Value>,
) -> EvalOutputWire {
    let EvalOutputFinalizeShared {
        input,
        input_latex,
        stored_id,
        strategy,
        steps_mode,
        steps,
        solve_steps,
        warnings,
        required_conditions,
        required_display,
        assumptions_used,
        blocked_hints,
        equivalence_diagnostics,
        budget_preset,
        strict,
        domain,
        timings_us,
        context_mode,
        branch_mode,
        expand_policy,
        complex_mode,
        const_fold,
        value_domain,
        complex_branch,
        inv_trig,
        assume_scope,
        ..
    } = shared;

    // Inline domain conditions into a bare `All real numbers` solve result (honest default surface).
    let (result, result_latex, result_chars) =
        match inline_all_reals_required_conditions(&payload.result, &required_display) {
            Some((text, latex)) => {
                let chars = text.chars().count();
                (text, Some(latex), chars)
            }
            None => (payload.result, payload.result_latex, payload.result_chars),
        };

    EvalOutputWire::from_build(EvalOutputWireBuild {
        input,
        input_latex,
        stored_id,
        result_chars,
        result,
        result_truncated: payload.result_truncated,
        result_latex,
        strategy,
        steps_mode,
        steps_count,
        steps,
        solve_steps,
        warnings,
        required_conditions,
        required_display,
        assumptions_used,
        blocked_hints,
        equivalence_diagnostics,
        budget_preset,
        strict,
        domain,
        stats: payload.stats,
        hash: payload.hash,
        timings_us,
        context_mode,
        branch_mode,
        expand_policy,
        complex_mode,
        const_fold,
        value_domain,
        complex_branch,
        inv_trig,
        assume_scope,
        wire,
    })
}
