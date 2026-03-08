use crate::SetCommandState;

use super::labels::{display_mode_label, on_off, steps_mode_label};

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
