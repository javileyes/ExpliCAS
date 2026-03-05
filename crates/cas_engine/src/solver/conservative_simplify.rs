//! Shared conservative simplify option presets for engine solver helpers.

/// Build simplify options for a specific domain mode.
pub(crate) fn simplify_options_for_domain(
    domain_mode: crate::DomainMode,
) -> crate::SimplifyOptions {
    cas_solver_core::conservative_eval_config::simplify_options_for_domain(domain_mode)
}

/// Conservative options used by bounded numeric-island folding passes.
pub(crate) fn conservative_numeric_fold_options() -> crate::SimplifyOptions {
    cas_solver_core::conservative_eval_config::conservative_numeric_fold_options()
}
