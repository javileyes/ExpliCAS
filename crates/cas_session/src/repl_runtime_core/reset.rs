use crate::{config::CasConfig, repl_core::ReplCore};

use super::build::build_simplifier;

/// Rebuild simplifier from persisted config and reset runtime/session state.
pub fn reset_repl_core_with_config(core: &mut ReplCore, config: &CasConfig) {
    cas_solver::session_api::lifecycle::reset_runtime_with_config(
        core,
        config,
        build_simplifier,
        crate::config::sync_simplifier_with_cas_config,
    );
}

/// Full reset: state reset + profile cache clear.
pub fn reset_repl_core_full_with_config(core: &mut ReplCore, config: &CasConfig) {
    cas_solver::session_api::lifecycle::reset_runtime_full_with_config(
        core,
        config,
        build_simplifier,
        crate::config::sync_simplifier_with_cas_config,
    );
}
