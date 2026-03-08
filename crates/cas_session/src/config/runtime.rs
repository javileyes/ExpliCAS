use super::{apply_solver_toggle_to_cas_config, solver_toggle_config_from_cas_config, CasConfig};
use cas_solver_core::config_runtime::ConfigCommandApplyContext;
use cas_solver_core::simplifier_config::SimplifierToggleConfig;

impl ConfigCommandApplyContext for CasConfig {
    fn current_toggles(&self) -> SimplifierToggleConfig {
        solver_toggle_config_from_cas_config(self)
    }

    fn save(&mut self) -> Result<(), String> {
        CasConfig::save(self).map_err(|error| error.to_string())
    }

    fn restore_defaults(&mut self) {
        *self = CasConfig::restore();
    }

    fn apply_toggles(&mut self, toggles: SimplifierToggleConfig) {
        apply_solver_toggle_to_cas_config(self, toggles);
    }
}
