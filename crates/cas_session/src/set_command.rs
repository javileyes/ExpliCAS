#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetDisplayMode {
    None,
    Succinct,
    Normal,
    Verbose,
}

/// Snapshot of current REPL `set` state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetCommandState {
    pub transform: bool,
    pub rationalize: cas_solver::AutoRationalizeLevel,
    pub heuristic_poly: cas_solver::HeuristicPoly,
    pub autoexpand_binomials: cas_solver::AutoExpandBinomials,
    pub steps_mode: cas_solver::StepsMode,
    pub display_mode: SetDisplayMode,
    pub max_rewrites: usize,
    pub debug_mode: bool,
}

/// Raw parsed `set` invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetCommandInput<'a> {
    ShowAll,
    ShowOption(&'a str),
    SetOption { option: &'a str, value: &'a str },
}

/// Normalized mutation plan for applying `set` changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetCommandPlan {
    pub set_transform: Option<bool>,
    pub set_rationalize: Option<cas_solver::AutoRationalizeLevel>,
    pub set_heuristic_poly: Option<cas_solver::HeuristicPoly>,
    pub set_autoexpand_binomials: Option<cas_solver::AutoExpandBinomials>,
    pub set_steps_mode: Option<cas_solver::StepsMode>,
    pub set_display_mode: Option<SetDisplayMode>,
    pub set_max_rewrites: Option<usize>,
    pub set_debug_mode: Option<bool>,
    pub message: String,
}

impl SetCommandPlan {
    fn with_message(message: impl Into<String>) -> Self {
        Self {
            set_transform: None,
            set_rationalize: None,
            set_heuristic_poly: None,
            set_autoexpand_binomials: None,
            set_steps_mode: None,
            set_display_mode: None,
            set_max_rewrites: None,
            set_debug_mode: None,
            message: message.into(),
        }
    }
}

/// Side effects produced while applying a `SetCommandPlan`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SetCommandApplyEffects {
    pub set_steps_mode: Option<cas_solver::StepsMode>,
    pub set_display_mode: Option<SetDisplayMode>,
}

/// Evaluated `set` command result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetCommandResult {
    ShowHelp { message: String },
    ShowValue { message: String },
    Apply { plan: SetCommandPlan },
    Invalid { message: String },
}

/// Parse raw `set ...` input.
pub fn parse_set_command_input(line: &str) -> SetCommandInput<'_> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() == 1 || (parts.len() == 2 && parts[1] == "show") {
        return SetCommandInput::ShowAll;
    }
    if parts.len() == 2 {
        return SetCommandInput::ShowOption(parts[1]);
    }
    SetCommandInput::SetOption {
        option: parts[1],
        value: parts[2],
    }
}

/// Evaluate a `set ...` command into a read/update decision.
pub fn evaluate_set_command_input(line: &str, state: SetCommandState) -> SetCommandResult {
    match parse_set_command_input(line) {
        SetCommandInput::ShowAll => SetCommandResult::ShowHelp {
            message: format_set_help_text(state),
        },
        SetCommandInput::ShowOption(option) => SetCommandResult::ShowValue {
            message: format_set_option_value(option, state),
        },
        SetCommandInput::SetOption { option, value } => evaluate_set_option(option, value, state),
    }
}

/// Apply a normalized `set` plan to runtime options.
///
/// Returns UI-adjacent effects (`steps_mode`/`display_mode`) so frontends can
/// update local renderer/runtime state without duplicating option mutation logic.
pub fn apply_set_command_plan(
    plan: &SetCommandPlan,
    simplify_options: &mut cas_solver::SimplifyOptions,
    eval_options: &mut cas_solver::EvalOptions,
    debug_mode: &mut bool,
) -> SetCommandApplyEffects {
    if let Some(enabled) = plan.set_transform {
        simplify_options.enable_transform = enabled;
    }
    if let Some(level) = plan.set_rationalize {
        simplify_options.rationalize.auto_level = level;
    }
    if let Some(mode) = plan.set_heuristic_poly {
        eval_options.shared.heuristic_poly = mode;
        simplify_options.shared.heuristic_poly = mode;
    }
    if let Some(mode) = plan.set_autoexpand_binomials {
        eval_options.shared.autoexpand_binomials = mode;
        simplify_options.shared.autoexpand_binomials = mode;
    }
    if let Some(max_rewrites) = plan.set_max_rewrites {
        simplify_options.budgets.max_total_rewrites = max_rewrites;
    }
    if let Some(value) = plan.set_debug_mode {
        *debug_mode = value;
    }
    if let Some(mode) = plan.set_steps_mode {
        eval_options.steps_mode = mode;
    }

    SetCommandApplyEffects {
        set_steps_mode: plan.set_steps_mode,
        set_display_mode: plan.set_display_mode,
    }
}

fn evaluate_set_option(option: &str, value: &str, state: SetCommandState) -> SetCommandResult {
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

/// Format a single `set <option>` value.
pub fn format_set_option_value(option: &str, state: SetCommandState) -> String {
    match option {
        "transform" => format!("transform: {}", on_off(state.transform)),
        "rationalize" => format!("rationalize: {:?}", state.rationalize),
        "heuristic_poly" => format!(
            "heuristic_poly: {}",
            on_off(state.heuristic_poly == cas_solver::HeuristicPoly::On)
        ),
        "autoexpand" | "autoexpand_binomials" => format!(
            "autoexpand: {}",
            on_off(state.autoexpand_binomials == cas_solver::AutoExpandBinomials::On)
        ),
        "max-rewrites" => format!("max-rewrites: {}", state.max_rewrites),
        "debug" => format!("debug: {}", on_off(state.debug_mode)),
        "steps" => format!(
            "steps: {} (display: {})",
            steps_mode_label(state.steps_mode),
            display_mode_label(state.display_mode)
        ),
        _ => format!(
            "Unknown option: {}\nUse 'set show' to see available options",
            option
        ),
    }
}

/// Format full `set show` help text with current values.
pub fn format_set_help_text(state: SetCommandState) -> String {
    let mut s = String::new();
    s.push_str("Pipeline settings:\n");
    s.push_str("  set transform <on|off>         Enable/disable distribution & expansion\n");
    s.push_str("  set rationalize <on|off|0|1|1.5>  Set rationalization level\n");
    s.push_str("  set heuristic_poly <on|off>    Smart polynomial simplification/factorization\n");
    s.push_str(
        "  set autoexpand <on|off>        Force expansion of binomial powers like (x+1)^n\n",
    );
    s.push_str("  set steps <on|off|...>         Step collection and display mode\n");
    s.push_str("  set max-rewrites <N>           Set max total rewrites (safety limit)\n");
    s.push_str("  set debug <on|off>             Show pipeline diagnostics after operations\n\n");
    s.push_str("Current settings:\n");
    s.push_str(&format!("  transform: {}\n", on_off(state.transform)));
    s.push_str(&format!("  rationalize: {:?}\n", state.rationalize));
    s.push_str(&format!(
        "  heuristic_poly: {}\n",
        on_off(state.heuristic_poly == cas_solver::HeuristicPoly::On)
    ));
    s.push_str(&format!(
        "  autoexpand: {}\n",
        on_off(state.autoexpand_binomials == cas_solver::AutoExpandBinomials::On)
    ));
    s.push_str(&format!(
        "  steps: {} (display: {})\n",
        steps_mode_label(state.steps_mode),
        display_mode_label(state.display_mode)
    ));
    s.push_str(&format!("  max-rewrites: {}\n", state.max_rewrites));
    s.push_str(&format!("  debug: {}", on_off(state.debug_mode)));
    s
}

fn on_off(enabled: bool) -> &'static str {
    if enabled {
        "on"
    } else {
        "off"
    }
}

fn steps_mode_label(mode: cas_solver::StepsMode) -> &'static str {
    match mode {
        cas_solver::StepsMode::On => "on",
        cas_solver::StepsMode::Off => "off",
        cas_solver::StepsMode::Compact => "compact",
    }
}

fn display_mode_label(mode: SetDisplayMode) -> &'static str {
    match mode {
        SetDisplayMode::None => "none",
        SetDisplayMode::Succinct => "succinct",
        SetDisplayMode::Normal => "normal",
        SetDisplayMode::Verbose => "verbose",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_set_command_plan, evaluate_set_command_input, format_set_help_text,
        parse_set_command_input, SetCommandApplyEffects, SetCommandInput, SetCommandResult,
        SetCommandState, SetDisplayMode,
    };

    fn state() -> SetCommandState {
        SetCommandState {
            transform: true,
            rationalize: cas_solver::AutoRationalizeLevel::Level15,
            heuristic_poly: cas_solver::HeuristicPoly::Off,
            autoexpand_binomials: cas_solver::AutoExpandBinomials::Off,
            steps_mode: cas_solver::StepsMode::On,
            display_mode: SetDisplayMode::Normal,
            max_rewrites: 200,
            debug_mode: false,
        }
    }

    #[test]
    fn parse_set_command_input_show_all() {
        assert_eq!(
            parse_set_command_input("set show"),
            SetCommandInput::ShowAll
        );
    }

    #[test]
    fn parse_set_command_input_set_option() {
        assert_eq!(
            parse_set_command_input("set transform off"),
            SetCommandInput::SetOption {
                option: "transform",
                value: "off",
            }
        );
    }

    #[test]
    fn evaluate_set_command_input_steps_verbose_sets_collection_and_display() {
        let out = evaluate_set_command_input("set steps verbose", state());
        match out {
            SetCommandResult::Apply { plan } => {
                assert_eq!(plan.set_steps_mode, Some(cas_solver::StepsMode::On));
                assert_eq!(plan.set_display_mode, Some(SetDisplayMode::Verbose));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn evaluate_set_command_input_invalid_max_rewrites_reports_usage() {
        let out = evaluate_set_command_input("set max-rewrites no", state());
        match out {
            SetCommandResult::Invalid { message } => {
                assert!(message.contains("Usage: set max-rewrites <number>"));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn format_set_help_text_includes_current_settings() {
        let text = format_set_help_text(state());
        assert!(text.contains("Current settings:"));
        assert!(text.contains("transform: on"));
        assert!(text.contains("steps: on (display: normal)"));
    }

    #[test]
    fn apply_set_command_plan_updates_states_and_effects() {
        let mut simplify_options = cas_solver::SimplifyOptions::default();
        let mut eval_options = cas_solver::EvalOptions::default();
        let mut debug_mode = false;

        let plan = super::SetCommandPlan {
            set_transform: Some(false),
            set_rationalize: Some(cas_solver::AutoRationalizeLevel::Level1),
            set_heuristic_poly: Some(cas_solver::HeuristicPoly::On),
            set_autoexpand_binomials: Some(cas_solver::AutoExpandBinomials::On),
            set_steps_mode: Some(cas_solver::StepsMode::Compact),
            set_display_mode: Some(SetDisplayMode::Succinct),
            set_max_rewrites: Some(123),
            set_debug_mode: Some(true),
            message: "ok".to_string(),
        };

        let effects = apply_set_command_plan(
            &plan,
            &mut simplify_options,
            &mut eval_options,
            &mut debug_mode,
        );

        assert_eq!(
            effects,
            SetCommandApplyEffects {
                set_steps_mode: Some(cas_solver::StepsMode::Compact),
                set_display_mode: Some(SetDisplayMode::Succinct),
            }
        );
        assert!(!simplify_options.enable_transform);
        assert_eq!(
            simplify_options.rationalize.auto_level,
            cas_solver::AutoRationalizeLevel::Level1
        );
        assert_eq!(
            simplify_options.shared.heuristic_poly,
            cas_solver::HeuristicPoly::On
        );
        assert_eq!(
            simplify_options.shared.autoexpand_binomials,
            cas_solver::AutoExpandBinomials::On
        );
        assert_eq!(simplify_options.budgets.max_total_rewrites, 123);
        assert_eq!(eval_options.steps_mode, cas_solver::StepsMode::Compact);
        assert_eq!(
            eval_options.shared.heuristic_poly,
            cas_solver::HeuristicPoly::On
        );
        assert_eq!(
            eval_options.shared.autoexpand_binomials,
            cas_solver::AutoExpandBinomials::On
        );
        assert!(debug_mode);
    }
}
