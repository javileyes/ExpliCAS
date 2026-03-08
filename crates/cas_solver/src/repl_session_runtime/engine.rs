use crate::profile_cache_command::ProfileCacheStore;
use crate::{
    evaluate_profile_cache_command_lines, evaluate_show_command_lines, ShowCommandContext,
};

use super::{ReplEngineRuntimeContext, ReplSessionEngineRuntimeContext};

/// Evaluate `show` command lines against runtime state/engine.
pub fn evaluate_show_command_lines_on_runtime<C>(
    context: &mut C,
    line: &str,
) -> Result<Vec<String>, String>
where
    C: ReplSessionEngineRuntimeContext,
    C::State: ShowCommandContext,
{
    context.with_engine_and_state(|engine, state| evaluate_show_command_lines(state, engine, line))
}

/// Evaluate profile `cache` command lines against runtime engine.
pub fn evaluate_profile_cache_command_lines_on_runtime<C: ReplEngineRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Vec<String>
where
    C::Engine: ProfileCacheStore,
{
    context.with_engine_mut(|engine| evaluate_profile_cache_command_lines(engine, line))
}
