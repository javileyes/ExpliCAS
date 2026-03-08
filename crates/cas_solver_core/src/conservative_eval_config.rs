//! Shared conservative eval-config presets used by runtime helpers.

use crate::domain_mode::DomainMode;
use crate::eval_config::EvalConfig;
use crate::phase_budgets::PhaseBudgets;
use crate::simplify_options::{SharedSemanticConfig, SimplifyOptions};
use crate::value_domain::ValueDomain;

/// Build a default eval config for a specific domain mode.
pub fn eval_config_for_domain(domain_mode: DomainMode) -> EvalConfig {
    EvalConfig {
        domain_mode,
        ..Default::default()
    }
}

/// Eval config used by conservative numeric-island folding passes.
pub fn conservative_numeric_fold_eval_config() -> EvalConfig {
    EvalConfig {
        domain_mode: DomainMode::Generic,
        value_domain: ValueDomain::RealOnly,
        ..Default::default()
    }
}

/// Build simplify options for a specific domain mode.
pub fn simplify_options_for_domain(domain_mode: DomainMode) -> SimplifyOptions {
    SimplifyOptions {
        shared: SharedSemanticConfig {
            semantics: eval_config_for_domain(domain_mode),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Conservative options used by bounded numeric-island folding passes.
pub fn conservative_numeric_fold_options() -> SimplifyOptions {
    SimplifyOptions {
        collect_steps: false,
        expand_mode: false,
        shared: SharedSemanticConfig {
            semantics: conservative_numeric_fold_eval_config(),
            ..Default::default()
        },
        budgets: PhaseBudgets {
            core_iters: crate::phase_budget_defaults::CONSERVATIVE_CORE_ITERS,
            transform_iters: crate::phase_budget_defaults::CONSERVATIVE_TRANSFORM_ITERS,
            rationalize_iters: crate::phase_budget_defaults::CONSERVATIVE_RATIONALIZE_ITERS,
            post_iters: crate::phase_budget_defaults::CONSERVATIVE_POST_ITERS,
            max_total_rewrites: crate::phase_budget_defaults::CONSERVATIVE_MAX_TOTAL_REWRITES,
        },
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        conservative_numeric_fold_eval_config, conservative_numeric_fold_options,
        eval_config_for_domain, simplify_options_for_domain,
    };
    use crate::domain_mode::DomainMode;
    use crate::value_domain::ValueDomain;

    #[test]
    fn eval_config_for_domain_sets_mode_and_keeps_defaults() {
        let cfg = eval_config_for_domain(DomainMode::Assume);
        assert_eq!(cfg.domain_mode, DomainMode::Assume);
        assert_eq!(cfg.value_domain, ValueDomain::RealOnly);
    }

    #[test]
    fn conservative_numeric_fold_eval_config_contract() {
        let cfg = conservative_numeric_fold_eval_config();
        assert_eq!(cfg.domain_mode, DomainMode::Generic);
        assert_eq!(cfg.value_domain, ValueDomain::RealOnly);
    }

    #[test]
    fn simplify_options_for_domain_sets_semantics_mode() {
        let opts = simplify_options_for_domain(DomainMode::Assume);
        assert_eq!(opts.shared.semantics.domain_mode, DomainMode::Assume);
    }

    #[test]
    fn conservative_numeric_fold_options_match_core_caps() {
        let opts = conservative_numeric_fold_options();
        assert!(!opts.collect_steps);
        assert!(!opts.expand_mode);
        assert_eq!(
            opts.budgets.core_iters,
            crate::phase_budget_defaults::CONSERVATIVE_CORE_ITERS
        );
        assert_eq!(
            opts.budgets.transform_iters,
            crate::phase_budget_defaults::CONSERVATIVE_TRANSFORM_ITERS
        );
        assert_eq!(
            opts.budgets.rationalize_iters,
            crate::phase_budget_defaults::CONSERVATIVE_RATIONALIZE_ITERS
        );
        assert_eq!(
            opts.budgets.post_iters,
            crate::phase_budget_defaults::CONSERVATIVE_POST_ITERS
        );
        assert_eq!(
            opts.budgets.max_total_rewrites,
            crate::phase_budget_defaults::CONSERVATIVE_MAX_TOTAL_REWRITES
        );
    }
}
