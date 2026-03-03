use crate::semantics_preset_catalog::{find_semantics_preset, semantics_presets};
use crate::semantics_preset_labels::{
    branch_policy_label, const_fold_mode_label, domain_mode_label, inverse_trig_policy_label,
    value_domain_label,
};
use crate::semantics_preset_types::{SemanticsPresetApplication, SemanticsPresetState};

pub fn format_semantics_preset_list_lines() -> Vec<String> {
    let mut lines = Vec::new();
    lines.push("Available presets:".to_string());
    for preset in semantics_presets() {
        lines.push(format!("  {:10} {}", preset.name, preset.description));
    }
    lines.push(String::new());
    lines.push("Usage:".to_string());
    lines.push("  semantics preset <name>       Apply preset".to_string());
    lines.push("  semantics preset help <name>  Show preset axes".to_string());
    lines
}

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

pub fn format_semantics_preset_application_lines(
    current: SemanticsPresetState,
    application: &SemanticsPresetApplication,
) -> Vec<String> {
    let next = application.next;
    let mut lines = Vec::new();
    lines.push(format!("Applied preset: {}", application.preset.name));
    lines.push("Changes:".to_string());

    let mut changes = 0;

    if current.domain != next.domain {
        lines.push(format!(
            "  domain_mode:  {} → {}",
            domain_mode_label(current.domain),
            domain_mode_label(next.domain)
        ));
        changes += 1;
    }

    if current.value != next.value {
        lines.push(format!(
            "  value_domain: {} → {}",
            value_domain_label(current.value),
            value_domain_label(next.value)
        ));
        changes += 1;
    }

    if current.branch != next.branch {
        lines.push(format!(
            "  branch:       {} → {}",
            branch_policy_label(current.branch),
            branch_policy_label(next.branch)
        ));
        changes += 1;
    }

    if current.inv_trig != next.inv_trig {
        lines.push(format!(
            "  inv_trig:     {} → {}",
            inverse_trig_policy_label(current.inv_trig),
            inverse_trig_policy_label(next.inv_trig)
        ));
        changes += 1;
    }

    if current.const_fold != next.const_fold {
        lines.push(format!(
            "  const_fold:   {} → {}",
            const_fold_mode_label(current.const_fold),
            const_fold_mode_label(next.const_fold)
        ));
        changes += 1;
    }

    if changes == 0 {
        lines.push("  (no changes - already at this preset)".to_string());
    }

    lines
}
