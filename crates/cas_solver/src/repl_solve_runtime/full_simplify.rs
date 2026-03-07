use crate::{EvalSession, SetDisplayMode};

use super::{display_mode, ReplSolveRuntimeContext};

pub(super) fn evaluate_full_simplify_command_lines_on_runtime<C: ReplSolveRuntimeContext>(
    context: &mut C,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<Vec<String>, String> {
    context.with_state_and_simplifier_mut(|state, simplifier| {
        let simplify_options = state.options().to_simplify_options();
        crate::evaluate_full_simplify_command_lines_with_resolver(
            simplifier,
            line,
            display_mode::map_full_simplify_display_mode(display_mode),
            simplify_options,
            |ctx, parsed_expr| {
                state
                    .resolve_all_with_diagnostics(ctx, parsed_expr)
                    .map(|(resolved_expr, _diag, _cache_hits)| resolved_expr)
                    .map_err(|error| error.to_string())
            },
        )
    })
}
