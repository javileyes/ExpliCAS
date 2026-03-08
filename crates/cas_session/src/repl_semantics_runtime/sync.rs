use crate::{CasConfig, ReplCore};

pub(super) fn sync_config_to_core(core: &mut ReplCore, config: &CasConfig) {
    crate::sync_simplifier_with_cas_config(core.simplifier_mut(), config)
}
