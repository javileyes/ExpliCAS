use super::ReplCore;
use crate::state_core::SessionState;
use cas_engine::{Engine, Simplifier};
use cas_solver_core::eval_options::EvalOptions;
use cas_solver_core::simplify_options::SimplifyOptions;

impl ReplCore {
    /// Run a closure with mutable access to engine + session state.
    pub fn with_engine_and_state<R>(
        &mut self,
        f: impl FnOnce(&mut Engine, &mut SessionState) -> R,
    ) -> R {
        f(&mut self.engine, &mut self.state)
    }

    /// Run a closure with mutable access to the inner simplifier.
    pub fn with_simplifier_mut<R>(&mut self, f: impl FnOnce(&mut Simplifier) -> R) -> R {
        f(&mut self.engine.simplifier)
    }

    /// Borrow session state + simplifier mutably in one operation.
    pub(crate) fn with_state_and_simplifier_mut<R>(
        &mut self,
        f: impl FnOnce(&mut SessionState, &mut Simplifier) -> R,
    ) -> R {
        f(&mut self.state, &mut self.engine.simplifier)
    }

    /// Borrow the session state.
    pub(crate) fn state(&self) -> &SessionState {
        &self.state
    }

    /// Borrow the session state mutably.
    pub(crate) fn state_mut(&mut self) -> &mut SessionState {
        &mut self.state
    }

    /// Borrow simplify pipeline options.
    pub(crate) fn simplify_options(&self) -> &SimplifyOptions {
        &self.simplify_options
    }

    /// Borrow simplify options + eval options mutably in one operation.
    pub(crate) fn with_simplify_and_eval_options_mut<R>(
        &mut self,
        f: impl FnOnce(&mut SimplifyOptions, &mut EvalOptions) -> R,
    ) -> R {
        f(&mut self.simplify_options, self.state.options_mut())
    }

    /// Borrow eval options from session state.
    pub(crate) fn eval_options(&self) -> &EvalOptions {
        self.state.options()
    }

    /// Borrow eval options from session state mutably.
    pub(crate) fn eval_options_mut(&mut self) -> &mut EvalOptions {
        self.state.options_mut()
    }

    /// Borrow the inner simplifier.
    pub(crate) fn simplifier(&self) -> &Simplifier {
        &self.engine.simplifier
    }

    /// Borrow the inner simplifier mutably.
    pub(crate) fn simplifier_mut(&mut self) -> &mut Simplifier {
        &mut self.engine.simplifier
    }
}
