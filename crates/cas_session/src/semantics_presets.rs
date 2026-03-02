/// Immutable semantics preset definition used by frontends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsPreset {
    pub name: &'static str,
    pub description: &'static str,
    pub domain: cas_engine::DomainMode,
    pub value: cas_engine::ValueDomain,
    pub branch: cas_engine::BranchPolicy,
    pub inv_trig: cas_engine::InverseTrigPolicy,
    pub const_fold: cas_engine::ConstFoldMode,
}

/// Runtime semantics state affected by preset application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsPresetState {
    pub domain: cas_engine::DomainMode,
    pub value: cas_engine::ValueDomain,
    pub branch: cas_engine::BranchPolicy,
    pub inv_trig: cas_engine::InverseTrigPolicy,
    pub const_fold: cas_engine::ConstFoldMode,
}

/// Build a preset-state snapshot from simplifier + eval options.
pub fn semantics_preset_state_from_options(
    simplify_options: &cas_engine::SimplifyOptions,
    eval_options: &cas_engine::EvalOptions,
) -> SemanticsPresetState {
    SemanticsPresetState {
        domain: simplify_options.shared.semantics.domain_mode,
        value: simplify_options.shared.semantics.value_domain,
        branch: simplify_options.shared.semantics.branch,
        inv_trig: simplify_options.shared.semantics.inv_trig,
        const_fold: eval_options.const_fold,
    }
}

/// Result of applying a semantics preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsPresetApplication {
    pub preset: SemanticsPreset,
    pub next: SemanticsPresetState,
}

/// End-to-end evaluation result for `semantics preset` command args.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticsPresetCommandOutput {
    pub lines: Vec<String>,
    pub applied: bool,
}

/// Error returned when preset application cannot proceed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticsPresetApplyError {
    UnknownPreset { name: String },
}

/// Apply preset state to both simplifier options and runtime eval options.
pub fn apply_semantics_preset_state_to_options(
    next: SemanticsPresetState,
    simplify_options: &mut cas_engine::SimplifyOptions,
    eval_options: &mut cas_engine::EvalOptions,
) {
    simplify_options.shared.semantics.domain_mode = next.domain;
    simplify_options.shared.semantics.value_domain = next.value;
    simplify_options.shared.semantics.branch = next.branch;
    simplify_options.shared.semantics.inv_trig = next.inv_trig;

    eval_options.shared.semantics.domain_mode = next.domain;
    eval_options.shared.semantics.value_domain = next.value;
    eval_options.shared.semantics.branch = next.branch;
    eval_options.shared.semantics.inv_trig = next.inv_trig;

    eval_options.const_fold = next.const_fold;
}

fn domain_mode_label(value: cas_engine::DomainMode) -> &'static str {
    match value {
        cas_engine::DomainMode::Strict => "strict",
        cas_engine::DomainMode::Generic => "generic",
        cas_engine::DomainMode::Assume => "assume",
    }
}

fn value_domain_label(value: cas_engine::ValueDomain) -> &'static str {
    match value {
        cas_engine::ValueDomain::RealOnly => "real",
        cas_engine::ValueDomain::ComplexEnabled => "complex",
    }
}

fn branch_policy_label(value: cas_engine::BranchPolicy) -> &'static str {
    match value {
        cas_engine::BranchPolicy::Principal => "principal",
    }
}

fn inverse_trig_policy_label(value: cas_engine::InverseTrigPolicy) -> &'static str {
    match value {
        cas_engine::InverseTrigPolicy::Strict => "strict",
        cas_engine::InverseTrigPolicy::PrincipalValue => "principal",
    }
}

fn const_fold_mode_label(value: cas_engine::ConstFoldMode) -> &'static str {
    match value {
        cas_engine::ConstFoldMode::Off => "off",
        cas_engine::ConstFoldMode::Safe => "safe",
    }
}

fn state_from_preset(preset: SemanticsPreset) -> SemanticsPresetState {
    SemanticsPresetState {
        domain: preset.domain,
        value: preset.value,
        branch: preset.branch,
        inv_trig: preset.inv_trig,
        const_fold: preset.const_fold,
    }
}

const SEMANTICS_PRESETS: [SemanticsPreset; 4] = [
    SemanticsPreset {
        name: "default",
        description: "Reset to engine defaults",
        domain: cas_engine::DomainMode::Generic,
        value: cas_engine::ValueDomain::RealOnly,
        branch: cas_engine::BranchPolicy::Principal,
        inv_trig: cas_engine::InverseTrigPolicy::Strict,
        const_fold: cas_engine::ConstFoldMode::Off,
    },
    SemanticsPreset {
        name: "strict",
        description: "Conservative real + strict domain",
        domain: cas_engine::DomainMode::Strict,
        value: cas_engine::ValueDomain::RealOnly,
        branch: cas_engine::BranchPolicy::Principal,
        inv_trig: cas_engine::InverseTrigPolicy::Strict,
        const_fold: cas_engine::ConstFoldMode::Off,
    },
    SemanticsPreset {
        name: "complex",
        description: "Enable ℂ + safe const_fold (sqrt(-1) → i)",
        domain: cas_engine::DomainMode::Generic,
        value: cas_engine::ValueDomain::ComplexEnabled,
        branch: cas_engine::BranchPolicy::Principal,
        inv_trig: cas_engine::InverseTrigPolicy::Strict,
        const_fold: cas_engine::ConstFoldMode::Safe,
    },
    SemanticsPreset {
        name: "school",
        description: "Real + principal inverse trig (arctan(tan(x)) → x)",
        domain: cas_engine::DomainMode::Generic,
        value: cas_engine::ValueDomain::RealOnly,
        branch: cas_engine::BranchPolicy::Principal,
        inv_trig: cas_engine::InverseTrigPolicy::PrincipalValue,
        const_fold: cas_engine::ConstFoldMode::Off,
    },
];

pub fn semantics_presets() -> &'static [SemanticsPreset] {
    &SEMANTICS_PRESETS
}

pub fn find_semantics_preset(name: &str) -> Option<SemanticsPreset> {
    SEMANTICS_PRESETS
        .iter()
        .copied()
        .find(|preset| preset.name == name)
}

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

pub fn apply_semantics_preset_by_name(
    name: &str,
) -> Result<SemanticsPresetApplication, SemanticsPresetApplyError> {
    let Some(preset) = find_semantics_preset(name) else {
        return Err(SemanticsPresetApplyError::UnknownPreset {
            name: name.to_string(),
        });
    };
    Ok(SemanticsPresetApplication {
        preset,
        next: state_from_preset(preset),
    })
}

/// Resolve and apply a semantics preset by name to runtime options.
pub fn apply_semantics_preset_by_name_to_options(
    name: &str,
    simplify_options: &mut cas_engine::SimplifyOptions,
    eval_options: &mut cas_engine::EvalOptions,
) -> Result<SemanticsPresetApplication, SemanticsPresetApplyError> {
    let application = apply_semantics_preset_by_name(name)?;
    apply_semantics_preset_state_to_options(application.next, simplify_options, eval_options);
    Ok(application)
}

/// Evaluate `semantics preset ...` args, mutating options on successful apply.
pub fn evaluate_semantics_preset_args_to_options(
    args: &[&str],
    simplify_options: &mut cas_engine::SimplifyOptions,
    eval_options: &mut cas_engine::EvalOptions,
) -> SemanticsPresetCommandOutput {
    match args.first().copied() {
        None => SemanticsPresetCommandOutput {
            lines: format_semantics_preset_list_lines(),
            applied: false,
        },
        Some("help") => SemanticsPresetCommandOutput {
            lines: format_semantics_preset_help_lines(args.get(1).copied()),
            applied: false,
        },
        Some(name) => {
            let current = semantics_preset_state_from_options(simplify_options, eval_options);
            match apply_semantics_preset_by_name_to_options(name, simplify_options, eval_options) {
                Ok(application) => SemanticsPresetCommandOutput {
                    lines: format_semantics_preset_application_lines(current, &application),
                    applied: true,
                },
                Err(SemanticsPresetApplyError::UnknownPreset { .. }) => {
                    SemanticsPresetCommandOutput {
                        lines: format_semantics_preset_help_lines(Some(name)),
                        applied: false,
                    }
                }
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::{
        apply_semantics_preset_by_name, apply_semantics_preset_by_name_to_options,
        apply_semantics_preset_state_to_options, evaluate_semantics_preset_args_to_options,
        find_semantics_preset, format_semantics_preset_application_lines,
        format_semantics_preset_help_lines, format_semantics_preset_list_lines,
        semantics_preset_state_from_options, SemanticsPresetApplyError,
        SemanticsPresetCommandOutput, SemanticsPresetState,
    };

    #[test]
    fn find_semantics_preset_returns_complex() {
        let preset = find_semantics_preset("complex").expect("preset exists");
        assert_eq!(preset.name, "complex");
    }

    #[test]
    fn format_semantics_preset_list_lines_contains_default() {
        let lines = format_semantics_preset_list_lines();
        assert!(lines.iter().any(|line| line.contains("default")));
    }

    #[test]
    fn format_semantics_preset_help_lines_unknown_includes_available_hint() {
        let lines = format_semantics_preset_help_lines(Some("missing"));
        assert!(lines
            .iter()
            .any(|line| line.contains("Available: default, strict, complex, school")));
    }

    #[test]
    fn apply_semantics_preset_by_name_unknown_returns_error() {
        let error = apply_semantics_preset_by_name("missing").expect_err("should fail");
        assert_eq!(
            error,
            SemanticsPresetApplyError::UnknownPreset {
                name: "missing".to_string(),
            }
        );
    }

    #[test]
    fn format_semantics_preset_application_lines_no_changes_reports_hint() {
        let application = apply_semantics_preset_by_name("default").expect("preset");
        let lines = format_semantics_preset_application_lines(application.next, &application);
        assert!(lines
            .iter()
            .any(|line| line.contains("(no changes - already at this preset)")));
    }

    #[test]
    fn format_semantics_preset_application_lines_reports_changed_axes() {
        let application = apply_semantics_preset_by_name("complex").expect("preset");
        let current = SemanticsPresetState {
            domain: cas_engine::DomainMode::Generic,
            value: cas_engine::ValueDomain::RealOnly,
            branch: cas_engine::BranchPolicy::Principal,
            inv_trig: cas_engine::InverseTrigPolicy::Strict,
            const_fold: cas_engine::ConstFoldMode::Off,
        };
        let lines = format_semantics_preset_application_lines(current, &application);
        assert!(lines
            .iter()
            .any(|line| line.contains("value_domain: real → complex")));
        assert!(lines
            .iter()
            .any(|line| line.contains("const_fold:   off → safe")));
    }

    #[test]
    fn apply_semantics_preset_state_to_options_updates_modes() {
        let mut simplify_options = cas_engine::SimplifyOptions::default();
        let mut eval_options = cas_engine::EvalOptions::default();
        let next = SemanticsPresetState {
            domain: cas_engine::DomainMode::Strict,
            value: cas_engine::ValueDomain::ComplexEnabled,
            branch: cas_engine::BranchPolicy::Principal,
            inv_trig: cas_engine::InverseTrigPolicy::PrincipalValue,
            const_fold: cas_engine::ConstFoldMode::Safe,
        };
        apply_semantics_preset_state_to_options(next, &mut simplify_options, &mut eval_options);
        assert_eq!(
            simplify_options.shared.semantics.domain_mode,
            cas_engine::DomainMode::Strict
        );
        assert_eq!(
            eval_options.shared.semantics.value_domain,
            cas_engine::ValueDomain::ComplexEnabled
        );
        assert_eq!(eval_options.const_fold, cas_engine::ConstFoldMode::Safe);
    }

    #[test]
    fn semantics_preset_state_from_options_reads_const_fold() {
        let simplify_options = cas_engine::SimplifyOptions::default();
        let eval_options = cas_engine::EvalOptions {
            const_fold: cas_engine::ConstFoldMode::Safe,
            ..cas_engine::EvalOptions::default()
        };
        let state = semantics_preset_state_from_options(&simplify_options, &eval_options);
        assert_eq!(state.const_fold, cas_engine::ConstFoldMode::Safe);
    }

    #[test]
    fn apply_semantics_preset_by_name_to_options_updates_runtime_state() {
        let mut simplify_options = cas_engine::SimplifyOptions::default();
        let mut eval_options = cas_engine::EvalOptions::default();
        let application = apply_semantics_preset_by_name_to_options(
            "complex",
            &mut simplify_options,
            &mut eval_options,
        )
        .expect("preset should exist");

        assert_eq!(application.preset.name, "complex");
        assert_eq!(
            simplify_options.shared.semantics.value_domain,
            cas_engine::ValueDomain::ComplexEnabled
        );
        assert_eq!(eval_options.const_fold, cas_engine::ConstFoldMode::Safe);
    }

    #[test]
    fn evaluate_semantics_preset_args_to_options_lists_when_empty() {
        let mut simplify_options = cas_engine::SimplifyOptions::default();
        let mut eval_options = cas_engine::EvalOptions::default();
        let out = evaluate_semantics_preset_args_to_options(
            &[],
            &mut simplify_options,
            &mut eval_options,
        );
        assert_eq!(
            out,
            SemanticsPresetCommandOutput {
                lines: format_semantics_preset_list_lines(),
                applied: false,
            }
        );
    }

    #[test]
    fn evaluate_semantics_preset_args_to_options_applies_known_preset() {
        let mut simplify_options = cas_engine::SimplifyOptions::default();
        let mut eval_options = cas_engine::EvalOptions::default();
        let out = evaluate_semantics_preset_args_to_options(
            &["complex"],
            &mut simplify_options,
            &mut eval_options,
        );
        assert!(out.applied);
        assert_eq!(
            simplify_options.shared.semantics.value_domain,
            cas_engine::ValueDomain::ComplexEnabled
        );
    }
}
