use crate::{
    clear_repl_profile_cache_on_runtime, reset_repl_runtime_state_on_runtime,
    ReplRuntimeStateContext, ReplSimplifierRuntimeContext, Simplifier,
};

/// Runtime context that can replace the active simplifier.
pub trait ReplConfiguredRuntimeContext:
    ReplRuntimeStateContext + ReplSimplifierRuntimeContext
{
    fn replace_simplifier(&mut self, simplifier: Simplifier);
}

/// Build a runtime context from config and synchronize simplifier toggles.
pub fn build_runtime_with_config<C, Config>(
    config: &Config,
    build_simplifier: impl FnOnce(&Config) -> Simplifier,
    build_context: impl FnOnce(Simplifier) -> C,
    sync_simplifier_with_config: impl FnOnce(&mut Simplifier, &Config),
) -> C
where
    C: ReplSimplifierRuntimeContext,
{
    let simplifier = build_simplifier(config);
    let mut context = build_context(simplifier);
    sync_simplifier_with_config(context.simplifier_mut(), config);
    context
}

/// Rebuild runtime simplifier from config and reset transient runtime/session state.
pub fn reset_runtime_with_config<C, Config>(
    context: &mut C,
    config: &Config,
    build_simplifier: impl FnOnce(&Config) -> Simplifier,
    sync_simplifier_with_config: impl FnOnce(&mut Simplifier, &Config),
) where
    C: ReplConfiguredRuntimeContext,
{
    context.replace_simplifier(build_simplifier(config));
    sync_simplifier_with_config(context.simplifier_mut(), config);
    reset_repl_runtime_state_on_runtime(context);
}

/// Full reset: reset runtime/session state and clear simplifier profile cache.
pub fn reset_runtime_full_with_config<C, Config>(
    context: &mut C,
    config: &Config,
    build_simplifier: impl FnOnce(&Config) -> Simplifier,
    sync_simplifier_with_config: impl FnOnce(&mut Simplifier, &Config),
) where
    C: ReplConfiguredRuntimeContext,
{
    reset_runtime_with_config(
        context,
        config,
        build_simplifier,
        sync_simplifier_with_config,
    );
    clear_repl_profile_cache_on_runtime(context);
}
