use crate::profile_cache_command::ProfileCacheStore;
use crate::{Engine, Simplifier};

/// Runtime context needed by session-centric REPL command adapters.
pub trait ReplSessionRuntimeContext {
    type State;

    fn state(&self) -> &Self::State;
}

/// Runtime context extension for commands that render via AST display context.
pub trait ReplSessionViewRuntimeContext: ReplSessionRuntimeContext {
    fn simplifier_context(&self) -> &cas_ast::Context;
}

/// Runtime context extension for commands that mutate session state directly.
pub trait ReplSessionStateMutRuntimeContext: ReplSessionRuntimeContext {
    fn state_mut(&mut self) -> &mut Self::State;
}

/// Runtime context extension for commands that need mutable session + simplifier.
pub trait ReplSessionSimplifierRuntimeContext: ReplSessionRuntimeContext {
    fn with_state_and_simplifier_mut<R>(
        &mut self,
        f: impl FnOnce(&mut Self::State, &mut Simplifier) -> R,
    ) -> R;
}

/// Runtime context extension for commands that only need direct `Engine` access.
pub trait ReplEngineRuntimeContext {
    type Engine: ProfileCacheStore;

    fn with_engine_mut<R>(&mut self, f: impl FnOnce(&mut Self::Engine) -> R) -> R;
}

/// Runtime context extension for session REPL commands that still require
/// direct `Engine` access.
pub trait ReplSessionEngineRuntimeContext: ReplSessionRuntimeContext {
    fn with_engine_and_state<R>(&mut self, f: impl FnOnce(&mut Engine, &mut Self::State) -> R)
        -> R;
}
