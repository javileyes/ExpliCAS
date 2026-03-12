use crate::{config::CasConfig, repl_core::ReplCore};

pub(super) fn sync_config_to_core(core: &mut ReplCore, config: &CasConfig) {
    crate::config::sync_simplifier_with_cas_config(core.simplifier_mut(), config)
}
