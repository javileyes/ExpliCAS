use super::{
    evaluate_autoexpand_command_on_runtime, evaluate_context_command_on_runtime,
    evaluate_semantics_command_on_runtime, ReplSemanticsRuntimeContext,
};

/// Evaluate `context` command and synchronize external config after simplifier rebuild.
pub fn evaluate_context_command_with_config_sync_on_runtime<C, Config>(
    line: &str,
    context: &mut C,
    config: &Config,
    sync_simplifier_with_config: impl FnOnce(&mut C, &Config),
) -> String
where
    C: ReplSemanticsRuntimeContext,
{
    evaluate_context_command_on_runtime(line, context, |context| {
        sync_simplifier_with_config(context, config)
    })
}

/// Evaluate `autoexpand` command and synchronize external config after simplifier rebuild.
pub fn evaluate_autoexpand_command_with_config_sync_on_runtime<C, Config>(
    line: &str,
    context: &mut C,
    config: &Config,
    sync_simplifier_with_config: impl FnOnce(&mut C, &Config),
) -> String
where
    C: ReplSemanticsRuntimeContext,
{
    evaluate_autoexpand_command_on_runtime(line, context, |context| {
        sync_simplifier_with_config(context, config)
    })
}

/// Evaluate `semantics` command and synchronize external config after simplifier rebuild.
pub fn evaluate_semantics_command_with_config_sync_on_runtime<C, Config>(
    line: &str,
    context: &mut C,
    config: &Config,
    sync_simplifier_with_config: impl FnOnce(&mut C, &Config),
) -> String
where
    C: ReplSemanticsRuntimeContext,
{
    evaluate_semantics_command_on_runtime(line, context, |context| {
        sync_simplifier_with_config(context, config)
    })
}
