use cas_ast::Context;
use cas_formatter::html_escape;
use cas_solver::{render_conditions_normalized, ImplicitCondition};

pub(super) fn normalize_global_requires(
    context: &mut Context,
    global_requires: &[ImplicitCondition],
) -> Vec<String> {
    if global_requires.is_empty() {
        return Vec::new();
    }

    render_conditions_normalized(context, global_requires)
        .iter()
        .map(|message| html_escape(message))
        .collect()
}
