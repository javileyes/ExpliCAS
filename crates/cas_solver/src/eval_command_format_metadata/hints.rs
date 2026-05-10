use crate::command_api::eval::EvalCommandEvalView;

pub(super) fn format_hint_lines(
    context: &mut cas_ast::Context,
    output: &EvalCommandEvalView,
    hints_enabled: bool,
    domain_mode: crate::DomainMode,
) -> Vec<String> {
    if !hints_enabled {
        return Vec::new();
    }

    let blocked_hint_result = match output.result {
        crate::EvalResult::Expr(expr) => expr,
        _ => output.resolved,
    };
    let required_conditions = output
        .diagnostics
        .requires
        .iter()
        .map(|item| item.cond.clone())
        .collect::<Vec<_>>();
    let hints = crate::filter_blocked_hints_for_eval(
        context,
        blocked_hint_result,
        &required_conditions,
        &output.blocked_hints,
    );
    if hints.is_empty() {
        Vec::new()
    } else {
        crate::format_eval_blocked_hints_lines(context, &hints, domain_mode)
    }
}
