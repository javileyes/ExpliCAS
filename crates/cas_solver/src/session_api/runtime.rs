//! Solver runtime adapters re-exported for session clients.

pub use crate::repl_runtime_configured::ReplConfiguredRuntimeContext;
pub use crate::repl_runtime_state::ReplRuntimeStateContext;
pub use crate::repl_session_runtime::{
    ReplEngineRuntimeContext, ReplSessionEngineRuntimeContext, ReplSessionRuntimeContext,
    ReplSessionSimplifierRuntimeContext, ReplSessionStateMutRuntimeContext,
    ReplSessionViewRuntimeContext,
};
