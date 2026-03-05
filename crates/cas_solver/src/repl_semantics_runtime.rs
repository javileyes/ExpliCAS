use crate::{EvalOptions, SemanticsCommandOutput, SimplifyOptions};

/// Result of applying a semantics-related command over runtime state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplSemanticsApplyOutput {
    pub message: String,
    pub rebuilt_simplifier: bool,
}

/// Runtime context needed by semantics command adapters.
pub trait ReplSemanticsRuntimeContext {
    fn eval_options_mut(&mut self) -> &mut EvalOptions;
    fn with_simplify_and_eval_options_mut<R>(
        &mut self,
        f: impl FnOnce(&mut SimplifyOptions, &mut EvalOptions) -> R,
    ) -> R;
    fn rebuild_simplifier_from_profile(&mut self);
}

/// Apply `context` command to runtime state.
pub fn apply_context_command_on_runtime<C: ReplSemanticsRuntimeContext>(
    line: &str,
    context: &mut C,
) -> ReplSemanticsApplyOutput {
    let applied = crate::evaluate_and_apply_context_command(line, context.eval_options_mut());
    if applied.rebuild_simplifier {
        context.rebuild_simplifier_from_profile();
    }
    ReplSemanticsApplyOutput {
        message: applied.message,
        rebuilt_simplifier: applied.rebuild_simplifier,
    }
}

/// Evaluate `context` command on runtime and optionally run a sync hook after rebuild.
pub fn evaluate_context_command_on_runtime<C: ReplSemanticsRuntimeContext>(
    line: &str,
    context: &mut C,
    on_rebuilt: impl FnOnce(&mut C),
) -> String {
    let applied = apply_context_command_on_runtime(line, context);
    if applied.rebuilt_simplifier {
        on_rebuilt(context);
    }
    applied.message
}

/// Apply `autoexpand` command to runtime state.
pub fn apply_autoexpand_command_on_runtime<C: ReplSemanticsRuntimeContext>(
    line: &str,
    context: &mut C,
) -> ReplSemanticsApplyOutput {
    let applied = crate::evaluate_and_apply_autoexpand_command(line, context.eval_options_mut());
    if applied.rebuild_simplifier {
        context.rebuild_simplifier_from_profile();
    }
    ReplSemanticsApplyOutput {
        message: applied.message,
        rebuilt_simplifier: applied.rebuild_simplifier,
    }
}

/// Evaluate `autoexpand` command on runtime and optionally run a sync hook after rebuild.
pub fn evaluate_autoexpand_command_on_runtime<C: ReplSemanticsRuntimeContext>(
    line: &str,
    context: &mut C,
    on_rebuilt: impl FnOnce(&mut C),
) -> String {
    let applied = apply_autoexpand_command_on_runtime(line, context);
    if applied.rebuilt_simplifier {
        on_rebuilt(context);
    }
    applied.message
}

/// Apply `semantics` command to runtime state.
pub fn apply_semantics_command_on_runtime<C: ReplSemanticsRuntimeContext>(
    line: &str,
    context: &mut C,
) -> SemanticsCommandOutput {
    let out = context.with_simplify_and_eval_options_mut(|simplify_options, eval_options| {
        crate::evaluate_semantics_command_line(line, simplify_options, eval_options)
    });
    if out.sync_simplifier {
        context.rebuild_simplifier_from_profile();
    }
    out
}

/// Evaluate `semantics` command on runtime and optionally run a sync hook after rebuild.
pub fn evaluate_semantics_command_on_runtime<C: ReplSemanticsRuntimeContext>(
    line: &str,
    context: &mut C,
    on_rebuilt: impl FnOnce(&mut C),
) -> String {
    let out = apply_semantics_command_on_runtime(line, context);
    if out.sync_simplifier {
        on_rebuilt(context);
    }
    out.lines.join("\n")
}

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
