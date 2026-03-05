use crate::eval_command_types::{EvalCommandEvalView, EvalMetadataLines};

pub(crate) fn format_eval_metadata_lines(
    context: &mut cas_ast::Context,
    output: &EvalCommandEvalView,
    requires_display: crate::RequiresDisplayLevel,
    debug_mode: bool,
    hints_enabled: bool,
    domain_mode: crate::DomainMode,
    assumption_reporting: crate::AssumptionReporting,
) -> EvalMetadataLines {
    let warning_lines = crate::format_domain_warning_lines(&output.domain_warnings, true, "⚠ ");

    let result_expr = match &output.result {
        crate::EvalResult::Expr(expr_id) => Some(*expr_id),
        _ => None,
    };
    let mut requires_lines = Vec::new();
    if !output.diagnostics.requires.is_empty() {
        let rendered = crate::format_diagnostics_requires_lines(
            context,
            &output.diagnostics,
            result_expr,
            requires_display,
            debug_mode,
        );
        if !rendered.is_empty() {
            requires_lines.push("ℹ️ Requires:".to_string());
            requires_lines.extend(rendered);
        }
    }

    let hint_lines = if hints_enabled {
        let hints =
            crate::filter_blocked_hints_for_eval(context, output.resolved, &output.blocked_hints);
        if hints.is_empty() {
            Vec::new()
        } else {
            crate::format_eval_blocked_hints_lines(context, &hints, domain_mode)
        }
    } else {
        Vec::new()
    };

    let assumption_lines = if assumption_reporting != crate::AssumptionReporting::Off {
        let assumed_conditions = crate::collect_assumed_conditions_from_steps(&output.steps);
        if assumed_conditions.is_empty() {
            Vec::new()
        } else {
            crate::format_assumed_conditions_report_lines(&assumed_conditions)
        }
    } else {
        Vec::new()
    };

    EvalMetadataLines {
        warning_lines,
        requires_lines,
        hint_lines,
        assumption_lines,
    }
}
