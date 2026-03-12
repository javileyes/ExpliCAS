use crate::repl_simplifier_runtime::ReplSimplifierRuntimeContext;
use cas_solver_core::config_runtime::ConfigCommandApplyOutput;

/// Evaluate and apply a config command on runtime state.
///
/// The caller injects how to apply config mutations and how to sync
/// simplifier toggles from that config type.
pub fn evaluate_and_apply_config_command_on_runtime<C, Config>(
    line: &str,
    config: &mut Config,
    context: &mut C,
    evaluate_and_apply: impl FnOnce(&str, &mut Config) -> ConfigCommandApplyOutput,
    sync_simplifier_with_config: impl FnOnce(&mut crate::Simplifier, &Config),
) -> String
where
    C: ReplSimplifierRuntimeContext,
{
    let applied = evaluate_and_apply(line, config);
    if applied.sync_simplifier {
        sync_simplifier_with_config(context.simplifier_mut(), config);
    }
    applied.message
}
