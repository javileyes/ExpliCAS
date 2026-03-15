use crate::semantics_preset_labels::{
    branch_policy_label, const_fold_mode_label, domain_mode_label, inverse_trig_policy_label,
    value_domain_label,
};
use crate::{SemanticsPresetApplication, SemanticsPresetState};

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
