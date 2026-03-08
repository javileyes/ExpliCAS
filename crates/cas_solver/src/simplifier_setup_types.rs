mod update;
pub use cas_solver_core::simplifier_config::{SimplifierRuleConfig, SimplifierToggleConfig};

/// Update one named toggle rule.
pub fn set_simplifier_toggle_rule(
    config: &mut SimplifierToggleConfig,
    rule: &str,
    enabled: bool,
) -> Result<(), String> {
    update::set_simplifier_toggle_rule(config, rule, enabled)
}
