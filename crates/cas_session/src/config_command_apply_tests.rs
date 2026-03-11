#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use cas_solver::session_api::{
        formatting::*, options::*, runtime::*, session_support::*, symbolic_commands::*, types::*,
    };

    #[test]
    fn evaluate_and_apply_config_command_updates_config_and_syncs() {
        let mut config = crate::CasConfig::default();
        let out = evaluate_and_apply_config_command("config enable distribute", &mut config);
        assert_eq!(
            out,
            ConfigCommandApplyOutput {
                message: "Rule 'distribute' set to true.".to_string(),
                sync_simplifier: true,
            }
        );
        assert!(config.distribute);
    }
}
