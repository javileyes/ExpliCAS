use crate::{EvalOptions, SimplifyOptions};
pub use cas_solver_core::repl_runtime::ReplSemanticsApplyOutput;

/// Runtime context needed by semantics command adapters.
pub trait ReplSemanticsRuntimeContext {
    fn eval_options_mut(&mut self) -> &mut EvalOptions;
    fn with_simplify_and_eval_options_mut<R>(
        &mut self,
        f: impl FnOnce(&mut SimplifyOptions, &mut EvalOptions) -> R,
    ) -> R;
    fn rebuild_simplifier_from_profile(&mut self);
}
