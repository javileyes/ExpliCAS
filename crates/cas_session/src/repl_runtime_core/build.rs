use crate::{CasConfig, ReplCore};
use cas_solver::session_api::options::build_simplifier_with_rule_config;

fn build_repl_simplifier_from_config(config: &CasConfig) -> cas_engine::Simplifier {
    build_simplifier_with_rule_config(crate::config::solver_rule_config_from_cas_config(config))
}

/// Build a `ReplCore` preconfigured from persisted CLI config.
pub fn build_repl_core_with_config(config: &CasConfig) -> ReplCore {
    cas_solver::build_runtime_with_config(
        config,
        build_repl_simplifier_from_config,
        ReplCore::with_simplifier,
        crate::config::sync_simplifier_with_cas_config,
    )
}

pub(super) fn build_simplifier(config: &CasConfig) -> cas_engine::Simplifier {
    build_repl_simplifier_from_config(config)
}
