use crate::repl_core::ReplCore;
use cas_engine::Simplifier;

impl ReplCore {
    /// Replace the inner simplifier.
    pub(crate) fn set_simplifier(&mut self, simplifier: Simplifier) {
        self.engine.simplifier = simplifier;
    }

    /// Rebuild simplifier from current eval profile.
    pub(crate) fn rebuild_simplifier_from_profile(&mut self) {
        self.engine.simplifier = Simplifier::with_profile(self.state.options());
    }

    /// Clear session state history/bindings.
    pub(crate) fn clear_state(&mut self) {
        self.state.clear();
    }
}
