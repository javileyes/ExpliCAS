use super::{apply_solver_toggle_to_cas_config, solver_toggle_config_from_cas_config, CasConfig};

impl cas_solver::ConfigCommandApplyContext for CasConfig {
    fn current_toggles(&self) -> cas_solver::SimplifierToggleConfig {
        solver_toggle_config_from_cas_config(self)
    }

    fn save(&mut self) -> Result<(), String> {
        CasConfig::save(self).map_err(|error| error.to_string())
    }

    fn restore_defaults(&mut self) {
        *self = CasConfig::restore();
    }

    fn apply_toggles(&mut self, toggles: cas_solver::SimplifierToggleConfig) {
        apply_solver_toggle_to_cas_config(self, toggles);
    }
}
