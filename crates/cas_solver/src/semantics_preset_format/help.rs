use crate::semantics_preset_catalog::find_semantics_preset;
use crate::semantics_preset_labels::{
    const_fold_mode_label, domain_mode_label, inverse_trig_policy_label, value_domain_label,
};

pub fn format_semantics_preset_help_lines(name: Option<&str>) -> Vec<String> {
    let mut lines = Vec::new();
    let Some(name) = name else {
        lines.push("Usage: semantics preset help <name>".to_string());
        lines.push("Presets: default, strict, complex, school".to_string());
        return lines;
    };

    if let Some(preset) = find_semantics_preset(name) {
        lines.push(format!("{}:", preset.name));
        lines.push(format!(
            "  domain_mode  = {}",
            domain_mode_label(preset.domain)
        ));
        lines.push(format!(
            "  value_domain = {}",
            value_domain_label(preset.value)
        ));
        lines.push("  branch       = principal".to_string());
        lines.push(format!(
            "  inv_trig     = {}",
            inverse_trig_policy_label(preset.inv_trig)
        ));
        lines.push(format!(
            "  const_fold   = {}",
            const_fold_mode_label(preset.const_fold)
        ));
        lines.push(String::new());
        lines.push(format!("Purpose: {}", preset.description));
    } else {
        lines.push(format!("Unknown preset: '{}'", name));
        lines.push("Available: default, strict, complex, school".to_string());
    }

    lines
}
