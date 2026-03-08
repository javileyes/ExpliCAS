use crate::simplifier_config::SimplifierToggleConfig;

/// Applied result for `config ...` command against mutable config state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigCommandApplyOutput {
    pub message: String,
    pub sync_simplifier: bool,
}

/// Context abstraction for applying config command effects.
pub trait ConfigCommandApplyContext {
    fn current_toggles(&self) -> SimplifierToggleConfig;
    fn save(&mut self) -> Result<(), String>;
    fn restore_defaults(&mut self);
    fn apply_toggles(&mut self, toggles: SimplifierToggleConfig);
}
