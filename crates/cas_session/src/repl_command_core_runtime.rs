//! Core REPL command adapters extracted from `repl_command_runtime`.

use crate::{ReplCore, SetDisplayMode};

/// Evaluate `equiv ...` against the active REPL simplifier.
pub fn evaluate_equiv_invocation_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_equiv_invocation_message(core.simplifier_mut(), line)
}

/// Evaluate `subst ...` against the active REPL simplifier.
pub fn evaluate_substitute_invocation_user_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    crate::evaluate_substitute_invocation_user_message(core.simplifier_mut(), line, display_mode)
}

/// Evaluate `solve ...` invocation against REPL core engine/session state.
pub fn evaluate_solve_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    let debug_mode = core.debug_mode();
    core.with_engine_and_state(|engine, state| {
        crate::evaluate_solve_command_message(engine, state, line, display_mode, debug_mode)
    })
}

/// Evaluate `simplify ...` invocation against REPL core simplifier/session state.
pub fn evaluate_full_simplify_command_lines_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<Vec<String>, String> {
    core.with_state_and_simplifier_mut(|state, simplifier| {
        crate::evaluate_full_simplify_command_lines(simplifier, state, line, display_mode)
    })
}

/// Evaluate `rationalize ...` invocation against REPL core simplifier.
pub fn evaluate_rationalize_command_lines_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<Vec<String>, String> {
    crate::evaluate_rationalize_command_lines(core.simplifier_mut(), line)
}

/// Refresh last health report using current REPL simplifier and health flag.
pub fn update_health_report_on_repl_core(core: &mut ReplCore) {
    core.set_last_health_report(crate::capture_health_report_if_enabled(
        core.simplifier(),
        core.health_enabled(),
    ));
}

/// Evaluate `health ...` command against REPL core and apply returned side-effects.
pub fn evaluate_health_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<String, String> {
    let last_stats = core.last_stats().cloned();
    let last_health_report = core.last_health_report().map(str::to_string);
    let out = crate::evaluate_health_command(
        core.simplifier_mut(),
        line,
        last_stats.as_ref(),
        last_health_report.as_deref(),
    )?;

    if let Some(enabled) = out.set_enabled {
        core.set_health_enabled(enabled);
    }
    if out.clear_last_report {
        core.clear_last_health_report();
    }

    Ok(out.lines.join("\n"))
}
