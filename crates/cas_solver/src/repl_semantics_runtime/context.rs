use crate::{EvalOptions, SimplifyOptions};

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
