use crate::health_command_messages::{
    health_clear_message, health_disable_message, health_enable_message,
};
use cas_solver_core::health_runtime::HealthCommandEvalOutput;

pub(super) fn build_show_last_output(lines: Vec<String>) -> HealthCommandEvalOutput {
    HealthCommandEvalOutput {
        lines,
        set_enabled: None,
        clear_last_report: false,
    }
}

pub(super) fn build_set_enabled_output(enabled: bool) -> HealthCommandEvalOutput {
    HealthCommandEvalOutput {
        lines: vec![if enabled {
            health_enable_message().to_string()
        } else {
            health_disable_message().to_string()
        }],
        set_enabled: Some(enabled),
        clear_last_report: false,
    }
}

pub(super) fn build_clear_output() -> HealthCommandEvalOutput {
    HealthCommandEvalOutput {
        lines: vec![health_clear_message().to_string()],
        set_enabled: None,
        clear_last_report: true,
    }
}

pub(super) fn build_status_output(lines: Vec<String>) -> HealthCommandEvalOutput {
    HealthCommandEvalOutput {
        lines,
        set_enabled: None,
        clear_last_report: false,
    }
}
