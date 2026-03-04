use cas_ast::Equation;

pub type SolverOptions = cas_solver_core::solver_options::SolverOptions;

/// Build solver options from engine eval options.
pub fn solver_options_from_eval_options(options: &crate::EvalOptions) -> SolverOptions {
    SolverOptions::from_axes(
        options.shared.semantics.value_domain,
        options.shared.semantics.domain_mode,
        options.shared.semantics.assume_scope,
        options.budget,
    )
}

/// Domain environment for solver operations.
///
/// Contains the "semantic ground" under which the solver operates:
/// - `required`: Constraints inferred from equation structure (e.g., sqrt(y) -> y >= 0)
///
/// This is passed explicitly rather than via TLS for clean reentrancy and testability.
pub(crate) type SolveDomainEnv =
    cas_solver_core::solve_aliases::SolveDomainEnv<crate::ImplicitDomain>;

/// Solver context — threaded explicitly through the solve pipeline.
///
/// Holds per-invocation state that was formerly stored in TLS,
/// enabling clean reentrancy for recursive/nested solves.
///
/// Shared sink state is centralized in `cas_solver_core::shared_context`,
/// so recursive sub-solves contribute to one accumulator set.
/// `solve_with_display_steps` creates one, and every recursive
/// `solve_with_ctx_and_options` / `solve_with_options` pushes into it.
pub type SolveCtx = cas_solver_core::solve_aliases::SolveCtx<
    SolveDomainEnv,
    crate::ImplicitCondition,
    crate::AssumptionEvent,
    cas_formatter::display_transforms::ScopeTag,
>;

// =============================================================================
// Type-Safe Step Pipeline (V2.9.8)
// =============================================================================
// These newtypes enforce that renderers only consume post-processed steps.
// This eliminates bifurcation between text/timeline outputs at compile time.

/// Display-ready solve steps after didactic cleanup and narration.
/// All renderers (text, timeline, JSON) consume this type only.
pub type DisplaySolveSteps = cas_solver_core::solve_aliases::DisplaySolveSteps<SolveStep>;

/// Diagnostics collected during solve operation.
///
/// This is returned alongside solutions to provide transparency about
/// what conditions were required vs assumed during solving.
pub type SolveDiagnostics = cas_solver_core::solve_aliases::SolveDiagnostics<
    crate::ImplicitCondition,
    crate::AssumptionEvent,
    crate::AssumptionRecord,
    cas_formatter::display_transforms::ScopeTag,
>;

/// Educational sub-step for solver derivations (e.g., completing the square)
/// Displayed as indented in REPL and collapsible in timeline.
pub type SolveSubStep =
    cas_solver_core::solve_aliases::SolveSubStep<Equation, crate::ImportanceLevel>;

pub type SolveStep =
    cas_solver_core::solve_aliases::SolveStep<Equation, crate::ImportanceLevel, SolveSubStep>;
