#[cfg(test)]
mod tests {
    use crate::repl_runtime_core::evaluate_and_apply_config_command_on_repl;

    #[test]
    fn evaluate_and_apply_config_command_on_repl_reports_status() {
        let mut config = crate::config::CasConfig::default();
        let mut core = crate::repl_core::ReplCore::new();
        let out =
            evaluate_and_apply_config_command_on_repl("config status", &mut config, &mut core);
        assert!(!out.trim().is_empty());
    }
}
