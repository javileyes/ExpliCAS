/// Canonical stateless session adapter for solver-owned eval orchestration.
pub type StatelessEvalSession = cas_session_core::eval::StatelessEvalSession<
    crate::EvalOptions,
    crate::DomainMode,
    crate::RequiredItem,
    crate::Step,
    crate::Diagnostics,
>;

/// Canonical store bound for solver-owned eval orchestration helpers.
pub trait SolverEvalStore:
    crate::EvalStore<
    DomainMode = crate::DomainMode,
    RequiredItem = crate::RequiredItem,
    Step = crate::Step,
    Diagnostics = crate::Diagnostics,
>
{
}

impl<T> SolverEvalStore for T where
    T: crate::EvalStore<
        DomainMode = crate::DomainMode,
        RequiredItem = crate::RequiredItem,
        Step = crate::Step,
        Diagnostics = crate::Diagnostics,
    >
{
}

/// Canonical session bound for solver-owned eval orchestration helpers.
pub trait SolverEvalSession:
    crate::EvalSession<
    Options = crate::EvalOptions,
    Diagnostics = crate::Diagnostics,
    Store: SolverEvalStore,
>
{
}

impl<T> SolverEvalSession for T where
    T: crate::EvalSession<
        Options = crate::EvalOptions,
        Diagnostics = crate::Diagnostics,
        Store: SolverEvalStore,
    >
{
}
