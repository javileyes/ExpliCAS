use crate::Simplifier;

/// Runtime context that exposes mutable access to the active simplifier.
pub trait ReplSimplifierRuntimeContext {
    fn simplifier_mut(&mut self) -> &mut Simplifier;
}
