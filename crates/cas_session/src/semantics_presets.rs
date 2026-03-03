pub use crate::semantics_preset_apply::{
    apply_semantics_preset_by_name, apply_semantics_preset_by_name_to_options,
    apply_semantics_preset_state_to_options, evaluate_semantics_preset_args_to_options,
    semantics_preset_state_from_options,
};
pub use crate::semantics_preset_catalog::{find_semantics_preset, semantics_presets};
pub use crate::semantics_preset_format::{
    format_semantics_preset_application_lines, format_semantics_preset_help_lines,
    format_semantics_preset_list_lines,
};
pub use crate::semantics_preset_types::{
    SemanticsPreset, SemanticsPresetApplication, SemanticsPresetApplyError,
    SemanticsPresetCommandOutput, SemanticsPresetState,
};

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
            domain: cas_solver::DomainMode::Generic,
            value: cas_solver::ValueDomain::RealOnly,
            branch: cas_solver::BranchPolicy::Principal,
            inv_trig: cas_solver::InverseTrigPolicy::Strict,
            const_fold: cas_solver::ConstFoldMode::Off,
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
        let mut simplify_options = cas_solver::SimplifyOptions::default();
        let mut eval_options = cas_solver::EvalOptions::default();
        let next = SemanticsPresetState {
            domain: cas_solver::DomainMode::Strict,
            value: cas_solver::ValueDomain::ComplexEnabled,
            branch: cas_solver::BranchPolicy::Principal,
            inv_trig: cas_solver::InverseTrigPolicy::PrincipalValue,
            const_fold: cas_solver::ConstFoldMode::Safe,
        };
        apply_semantics_preset_state_to_options(next, &mut simplify_options, &mut eval_options);
        assert_eq!(
            simplify_options.shared.semantics.domain_mode,
            cas_solver::DomainMode::Strict
        );
        assert_eq!(
            eval_options.shared.semantics.value_domain,
            cas_solver::ValueDomain::ComplexEnabled
        );
        assert_eq!(eval_options.const_fold, cas_solver::ConstFoldMode::Safe);
    }

    #[test]
    fn semantics_preset_state_from_options_reads_const_fold() {
        let simplify_options = cas_solver::SimplifyOptions::default();
        let eval_options = cas_solver::EvalOptions {
            const_fold: cas_solver::ConstFoldMode::Safe,
            ..cas_solver::EvalOptions::default()
        };
        let state = semantics_preset_state_from_options(&simplify_options, &eval_options);
        assert_eq!(state.const_fold, cas_solver::ConstFoldMode::Safe);
    }

    #[test]
    fn apply_semantics_preset_by_name_to_options_updates_runtime_state() {
        let mut simplify_options = cas_solver::SimplifyOptions::default();
        let mut eval_options = cas_solver::EvalOptions::default();
        let application = apply_semantics_preset_by_name_to_options(
            "complex",
            &mut simplify_options,
            &mut eval_options,
        )
        .expect("preset should exist");

        assert_eq!(application.preset.name, "complex");
        assert_eq!(
            simplify_options.shared.semantics.value_domain,
            cas_solver::ValueDomain::ComplexEnabled
        );
        assert_eq!(eval_options.const_fold, cas_solver::ConstFoldMode::Safe);
    }

    #[test]
    fn evaluate_semantics_preset_args_to_options_lists_when_empty() {
        let mut simplify_options = cas_solver::SimplifyOptions::default();
        let mut eval_options = cas_solver::EvalOptions::default();
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
        let mut simplify_options = cas_solver::SimplifyOptions::default();
        let mut eval_options = cas_solver::EvalOptions::default();
        let out = evaluate_semantics_preset_args_to_options(
            &["complex"],
            &mut simplify_options,
            &mut eval_options,
        );
        assert!(out.applied);
        assert_eq!(
            simplify_options.shared.semantics.value_domain,
            cas_solver::ValueDomain::ComplexEnabled
        );
    }
}
