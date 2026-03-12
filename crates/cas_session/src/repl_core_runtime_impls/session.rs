use crate::{repl_core::ReplCore, state_core::SessionState};
use cas_solver::runtime::{Engine, Simplifier};
use cas_solver::session_api::runtime::{
    ReplEngineRuntimeContext, ReplSessionEngineRuntimeContext, ReplSessionRuntimeContext,
    ReplSessionSimplifierRuntimeContext, ReplSessionStateMutRuntimeContext,
    ReplSessionViewRuntimeContext,
};

impl ReplSessionRuntimeContext for ReplCore {
    type State = SessionState;

    fn state(&self) -> &Self::State {
        ReplCore::state(self)
    }
}

impl ReplSessionViewRuntimeContext for ReplCore {
    fn simplifier_context(&self) -> &cas_ast::Context {
        &ReplCore::simplifier(self).context
    }
}

impl ReplSessionStateMutRuntimeContext for ReplCore {
    fn state_mut(&mut self) -> &mut Self::State {
        ReplCore::state_mut(self)
    }
}

impl ReplSessionSimplifierRuntimeContext for ReplCore {
    fn with_state_and_simplifier_mut<R>(
        &mut self,
        f: impl FnOnce(&mut Self::State, &mut Simplifier) -> R,
    ) -> R {
        ReplCore::with_state_and_simplifier_mut(self, f)
    }
}

impl ReplSessionEngineRuntimeContext for ReplCore {
    fn with_engine_and_state<R>(
        &mut self,
        f: impl FnOnce(&mut Engine, &mut Self::State) -> R,
    ) -> R {
        ReplCore::with_engine_and_state(self, f)
    }
}

impl ReplEngineRuntimeContext for ReplCore {
    type Engine = Engine;

    fn with_engine_mut<R>(&mut self, f: impl FnOnce(&mut Self::Engine) -> R) -> R {
        ReplCore::with_engine_and_state(self, |engine, _state| f(engine))
    }
}
