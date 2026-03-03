//! Session-centric runtime adapters extracted from `repl_command_runtime`.

use crate::ReplCore;

/// Render `vars` command output using REPL core state/context.
pub fn evaluate_vars_command_message_on_repl_core(core: &ReplCore) -> String {
    crate::evaluate_vars_command_lines_with_context(core.state(), &core.simplifier().context)
        .join("\n")
}

/// Render `history` command output using REPL core state/context.
pub fn evaluate_history_command_message_on_repl_core(core: &ReplCore) -> String {
    crate::evaluate_history_command_lines_with_context(core.state(), &core.simplifier().context)
        .join("\n")
}

/// Evaluate `show` command lines against REPL core state/engine.
pub fn evaluate_show_command_lines_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<Vec<String>, String> {
    core.with_engine_and_state(|engine, state| {
        crate::evaluate_show_command_lines(state, engine, line)
    })
}

/// Evaluate `clear` command lines against REPL core state.
pub fn evaluate_clear_command_lines_on_repl_core(core: &mut ReplCore, line: &str) -> Vec<String> {
    crate::evaluate_clear_command_lines(core.state_mut(), line)
}

/// Evaluate `del` command message against REPL core state.
pub fn evaluate_delete_history_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> String {
    crate::evaluate_delete_history_command_message(core.state_mut(), line)
}

/// Evaluate profile `cache` command lines against REPL core engine.
pub fn evaluate_profile_cache_command_lines_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Vec<String> {
    crate::evaluate_profile_cache_command_lines(core.engine_mut(), line)
}

/// Evaluate `budget ...` command message against REPL core session state.
pub fn evaluate_solve_budget_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> String {
    crate::evaluate_solve_budget_command_message(core.state_mut(), line)
}

/// Evaluate `let ...` command against REPL core and return user-facing message.
pub fn evaluate_let_assignment_command_message_on_repl_core(
    core: &mut ReplCore,
    input: &str,
) -> Result<String, String> {
    core.with_state_and_simplifier_mut(|state, simplifier| {
        crate::evaluate_let_assignment_command_message_with_simplifier(state, simplifier, input)
    })
}

/// Evaluate assignment command against REPL core and return user-facing message.
pub fn evaluate_assignment_command_message_on_repl_core(
    core: &mut ReplCore,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<String, String> {
    core.with_state_and_simplifier_mut(|state, simplifier| {
        crate::evaluate_assignment_command_message_with_simplifier(
            state, simplifier, name, expr_str, lazy,
        )
    })
}
