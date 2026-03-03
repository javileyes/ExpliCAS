#[cfg(test)]
mod tests {
    use crate::{
        format_semantics_axis_lines, format_semantics_overview_lines,
        format_semantics_unknown_subcommand_message, semantics_help_message,
        semantics_view_state_from_options, SemanticsViewState,
    };

    fn state() -> SemanticsViewState {
        SemanticsViewState {
            domain_mode: cas_solver::DomainMode::Generic,
            value_domain: cas_solver::ValueDomain::RealOnly,
            branch: cas_solver::BranchPolicy::Principal,
            inv_trig: cas_solver::InverseTrigPolicy::Strict,
            const_fold: cas_solver::ConstFoldMode::Off,
            assumption_reporting: cas_solver::AssumptionReporting::Summary,
            assume_scope: cas_solver::AssumeScope::Real,
            hints_enabled: true,
            requires_display: cas_solver::RequiresDisplayLevel::Essential,
        }
    }

    #[test]
    fn format_semantics_overview_lines_marks_inactive_branch_for_real_mode() {
        let lines = format_semantics_overview_lines(&state());
        assert!(lines
            .iter()
            .any(|line| line.contains("branch: principal (inactive: value_domain=real)")));
    }

    #[test]
    fn format_semantics_axis_lines_reports_unknown_axis() {
        let lines = format_semantics_axis_lines(&state(), "missing");
        assert_eq!(lines, vec!["Unknown axis: missing".to_string()]);
    }

    #[test]
    fn semantics_help_message_mentions_presets() {
        assert!(semantics_help_message().contains("semantics preset"));
    }

    #[test]
    fn format_semantics_unknown_subcommand_message_mentions_usage() {
        let text = format_semantics_unknown_subcommand_message("nope");
        assert!(text.contains("Unknown semantics subcommand: 'nope'"));
        assert!(text.contains("Usage: semantics"));
    }

    #[test]
    fn semantics_view_state_from_options_reads_requires_display() {
        let simplify_options = cas_solver::SimplifyOptions::default();
        let eval_options = cas_solver::EvalOptions {
            requires_display: cas_solver::RequiresDisplayLevel::All,
            ..cas_solver::EvalOptions::default()
        };
        let state = semantics_view_state_from_options(&simplify_options, &eval_options);
        assert_eq!(
            state.requires_display,
            cas_solver::RequiresDisplayLevel::All
        );
    }
}
