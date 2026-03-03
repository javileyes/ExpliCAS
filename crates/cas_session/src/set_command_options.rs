use crate::{
    format_set_help_text, SetCommandPlan, SetCommandResult, SetCommandState, SetDisplayMode,
};

pub(crate) fn evaluate_set_option(
    option: &str,
    value: &str,
    state: SetCommandState,
) -> SetCommandResult {
    match option {
        "transform" => match value {
            "on" | "true" | "1" => {
                let mut plan = SetCommandPlan::with_message(
                    "Transform phase ENABLED (distribution, expansion)",
                );
                plan.set_transform = Some(true);
                SetCommandResult::Apply { plan }
            }
            "off" | "false" | "0" => {
                let mut plan = SetCommandPlan::with_message(
                    "Transform phase DISABLED (no distribution/expansion)",
                );
                plan.set_transform = Some(false);
                SetCommandResult::Apply { plan }
            }
            _ => SetCommandResult::Invalid {
                message: "Usage: set transform <on|off>".to_string(),
            },
        },
        "autoexpand" | "autoexpand_binomials" => match value {
            "on" | "true" | "1" => {
                let mut plan = SetCommandPlan::with_message(
                    "Autoexpand binomials: ON (always expand)\n  (x+1)^5 will now expand to x⁵+5x⁴+10x³+10x²+5x+1",
                );
                plan.set_autoexpand_binomials = Some(cas_solver::AutoExpandBinomials::On);
                SetCommandResult::Apply { plan }
            }
            "off" | "false" | "0" => {
                let mut plan = SetCommandPlan::with_message(
                    "Autoexpand binomials: OFF (default, keep factored form)",
                );
                plan.set_autoexpand_binomials = Some(cas_solver::AutoExpandBinomials::Off);
                SetCommandResult::Apply { plan }
            }
            _ => SetCommandResult::Invalid {
                message: "Usage: set autoexpand <off|on>".to_string(),
            },
        },
        "heuristic_poly" => match value {
            "on" | "true" | "1" => {
                let mut plan = SetCommandPlan::with_message(
                    "Heuristic polynomial simplification: ON\n  - Extract common factors in Add/Sub\n  - Poly normalize if no factor found\n  Example: (x+1)^4 + 4·(x+1)^3 → (x+1)³·(x+5)",
                );
                plan.set_heuristic_poly = Some(cas_solver::HeuristicPoly::On);
                SetCommandResult::Apply { plan }
            }
            "off" | "false" | "0" => {
                let mut plan = SetCommandPlan::with_message(
                    "Heuristic polynomial simplification: OFF (default)",
                );
                plan.set_heuristic_poly = Some(cas_solver::HeuristicPoly::Off);
                SetCommandResult::Apply { plan }
            }
            _ => SetCommandResult::Invalid {
                message: "Usage: set heuristic_poly <off|on>".to_string(),
            },
        },
        "rationalize" => match value {
            "on" | "true" | "auto" => {
                let mut plan = SetCommandPlan::with_message("Rationalization ENABLED (Level 1.5)");
                plan.set_rationalize = Some(cas_solver::AutoRationalizeLevel::Level15);
                SetCommandResult::Apply { plan }
            }
            "off" | "false" => {
                let mut plan = SetCommandPlan::with_message("Rationalization DISABLED");
                plan.set_rationalize = Some(cas_solver::AutoRationalizeLevel::Off);
                SetCommandResult::Apply { plan }
            }
            "0" | "level0" => {
                let mut plan =
                    SetCommandPlan::with_message("Rationalization set to Level 0 (single sqrt)");
                plan.set_rationalize = Some(cas_solver::AutoRationalizeLevel::Level0);
                SetCommandResult::Apply { plan }
            }
            "1" | "level1" => {
                let mut plan = SetCommandPlan::with_message(
                    "Rationalization set to Level 1 (binomial conjugate)",
                );
                plan.set_rationalize = Some(cas_solver::AutoRationalizeLevel::Level1);
                SetCommandResult::Apply { plan }
            }
            "1.5" | "level15" => {
                let mut plan = SetCommandPlan::with_message(
                    "Rationalization set to Level 1.5 (same-surd products)",
                );
                plan.set_rationalize = Some(cas_solver::AutoRationalizeLevel::Level15);
                SetCommandResult::Apply { plan }
            }
            _ => SetCommandResult::Invalid {
                message: "Usage: set rationalize <on|off|0|1|1.5>".to_string(),
            },
        },
        "max-rewrites" => match value.parse::<usize>() {
            Ok(n) => {
                let mut plan = SetCommandPlan::with_message(format!("Max rewrites set to {}", n));
                plan.set_max_rewrites = Some(n);
                SetCommandResult::Apply { plan }
            }
            Err(_) => SetCommandResult::Invalid {
                message: "Usage: set max-rewrites <number>".to_string(),
            },
        },
        "steps" => evaluate_set_steps(value),
        "debug" => match value {
            "on" | "true" | "1" => {
                let mut plan = SetCommandPlan::with_message(
                    "Debug mode ENABLED (pipeline diagnostics after each operation)",
                );
                plan.set_debug_mode = Some(true);
                SetCommandResult::Apply { plan }
            }
            "off" | "false" | "0" => {
                let mut plan = SetCommandPlan::with_message("Debug mode DISABLED");
                plan.set_debug_mode = Some(false);
                SetCommandResult::Apply { plan }
            }
            _ => SetCommandResult::Invalid {
                message: "Usage: set debug <on|off>".to_string(),
            },
        },
        _ => SetCommandResult::ShowHelp {
            message: format_set_help_text(state),
        },
    }
}

fn evaluate_set_steps(value: &str) -> SetCommandResult {
    match value {
        "on" => {
            let mut plan =
                SetCommandPlan::with_message("Steps: on (full collection, normal display)");
            plan.set_steps_mode = Some(cas_solver::StepsMode::On);
            plan.set_display_mode = Some(SetDisplayMode::Normal);
            SetCommandResult::Apply { plan }
        }
        "off" => {
            let mut plan = SetCommandPlan::with_message("Steps: off");
            plan.set_steps_mode = Some(cas_solver::StepsMode::Off);
            plan.set_display_mode = Some(SetDisplayMode::None);
            SetCommandResult::Apply { plan }
        }
        "compact" => {
            let mut plan =
                SetCommandPlan::with_message("Steps: compact (no before/after snapshots)");
            plan.set_steps_mode = Some(cas_solver::StepsMode::Compact);
            SetCommandResult::Apply { plan }
        }
        "verbose" => {
            let mut plan = SetCommandPlan::with_message("Steps: verbose (all rules, full detail)");
            plan.set_steps_mode = Some(cas_solver::StepsMode::On);
            plan.set_display_mode = Some(SetDisplayMode::Verbose);
            SetCommandResult::Apply { plan }
        }
        "succinct" => {
            let mut plan =
                SetCommandPlan::with_message("Steps: succinct (compact 1-line per step)");
            plan.set_steps_mode = Some(cas_solver::StepsMode::On);
            plan.set_display_mode = Some(SetDisplayMode::Succinct);
            SetCommandResult::Apply { plan }
        }
        "normal" => {
            let mut plan = SetCommandPlan::with_message("Steps: normal (default display)");
            plan.set_steps_mode = Some(cas_solver::StepsMode::On);
            plan.set_display_mode = Some(SetDisplayMode::Normal);
            SetCommandResult::Apply { plan }
        }
        "none" => {
            let mut plan =
                SetCommandPlan::with_message("Steps display: none (collection still active)");
            plan.set_display_mode = Some(SetDisplayMode::None);
            SetCommandResult::Apply { plan }
        }
        _ => SetCommandResult::Invalid {
            message: "Usage: set steps <on|off|compact|verbose|succinct|normal|none>".to_string(),
        },
    }
}
