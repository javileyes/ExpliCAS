use cas_solver_core::eval_options::EvalOptions;

use crate::env::Environment;
use crate::state_eval_store::SessionEvalStore;

mod constructors;
mod runtime;
mod snapshot;

/// Bundled session state for portability (CLI/Web/FFI).
///
/// This crate-local type is the migration target for Phase 3.
/// `cas_engine` remains stateless and consumes it via the shared `EvalSession` trait.
#[derive(Default, Debug)]
pub struct SessionState {
    pub(crate) store: SessionEvalStore,
    pub(crate) env: Environment,
    pub(crate) options: EvalOptions,
}
