use crate::{SetCommandState, SetDisplayMode};

/// Format a single `set <option>` value.
pub fn format_set_option_value(option: &str, state: SetCommandState) -> String {
    match option {
        "transform" => format!("transform: {}", on_off(state.transform)),
        "rationalize" => format!("rationalize: {:?}", state.rationalize),
        "heuristic_poly" => format!(
            "heuristic_poly: {}",
            on_off(state.heuristic_poly == crate::HeuristicPoly::On)
        ),
        "autoexpand" | "autoexpand_binomials" => format!(
            "autoexpand: {}",
            on_off(state.autoexpand_binomials == crate::AutoExpandBinomials::On)
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
        on_off(state.heuristic_poly == crate::HeuristicPoly::On)
    ));
    s.push_str(&format!(
        "  autoexpand: {}\n",
        on_off(state.autoexpand_binomials == crate::AutoExpandBinomials::On)
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

fn steps_mode_label(mode: crate::StepsMode) -> &'static str {
    match mode {
        crate::StepsMode::On => "on",
        crate::StepsMode::Off => "off",
        crate::StepsMode::Compact => "compact",
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
