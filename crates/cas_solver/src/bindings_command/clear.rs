use cas_solver_core::session_runtime::ClearBindingsResult;

use super::BindingsContext;

/// Apply a `clear` command over session bindings.
///
/// Accepts either:
/// - `clear` (clear all bindings)
/// - `clear <name> [<name> ...]` (clear selected bindings)
pub fn clear_bindings_command<C: BindingsContext>(
    context: &mut C,
    input: &str,
) -> ClearBindingsResult {
    let trimmed = input.trim();
    let all_mode = trimmed == "clear" || trimmed.is_empty();

    if all_mode {
        let cleared_count = context.binding_count();
        context.clear_bindings();
        return ClearBindingsResult::All { cleared_count };
    }

    let names_part = trimmed.strip_prefix("clear").unwrap_or(trimmed).trim();
    let names: Vec<&str> = names_part.split_whitespace().collect();

    let mut cleared_count = 0;
    let mut missing_names = Vec::new();
    for name in names {
        if context.unset_binding(name) {
            cleared_count += 1;
        } else {
            missing_names.push(name.to_string());
        }
    }

    ClearBindingsResult::Selected {
        cleared_count,
        missing_names,
    }
}
