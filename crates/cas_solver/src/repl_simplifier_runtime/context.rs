use crate::Simplifier;

/// Runtime context that exposes mutable access to the active simplifier.
pub trait ReplSimplifierRuntimeContext {
    fn simplifier_mut(&mut self) -> &mut Simplifier;

    /// The session's value domain (`semantics set value ...`), consumed by the
    /// commands whose comparators run PLAIN `simplify()` on a shared
    /// simplifier (`equiv`): they arm the simplifier's sticky value domain
    /// from it so the comparison honors the session axis (audit 2026-07-30,
    /// ficha S5-002 — `equiv((e^z)^w, e^(z*w))` confirmed a real-only
    /// identity under `semantics set value complex`). RealOnly default keeps
    /// every context that predates the axis byte-identical.
    fn session_value_domain(&self) -> cas_engine::ValueDomain {
        cas_engine::ValueDomain::RealOnly
    }
}
