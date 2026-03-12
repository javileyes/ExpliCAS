#[cfg(test)]
mod tests {
    use cas_solver_core::rationalize_policy::AutoRationalizeLevel;

    use crate::{
        apply_set_command_plan, evaluate_set_command_input, format_set_help_text,
        parse_set_command_input, SetCommandApplyEffects, SetCommandInput, SetCommandResult,
        SetCommandState, SetDisplayMode,
    };

    fn state() -> SetCommandState {
        SetCommandState {
            transform: true,
            rationalize: AutoRationalizeLevel::Level15,
            heuristic_poly: crate::HeuristicPoly::Off,
            autoexpand_binomials: crate::AutoExpandBinomials::Off,
            steps_mode: crate::StepsMode::On,
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
                assert_eq!(plan.set_steps_mode, Some(crate::StepsMode::On));
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
        let mut simplify_options = crate::SimplifyOptions::default();
        let mut eval_options = crate::EvalOptions::default();
        let mut debug_mode = false;

        let plan = crate::SetCommandPlan {
            set_transform: Some(false),
            set_rationalize: Some(AutoRationalizeLevel::Level1),
            set_heuristic_poly: Some(crate::HeuristicPoly::On),
            set_autoexpand_binomials: Some(crate::AutoExpandBinomials::On),
            set_steps_mode: Some(crate::StepsMode::Compact),
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
                set_steps_mode: Some(crate::StepsMode::Compact),
                set_display_mode: Some(SetDisplayMode::Succinct),
            }
        );
        assert!(!simplify_options.enable_transform);
        assert_eq!(
            simplify_options.rationalize.auto_level,
            AutoRationalizeLevel::Level1
        );
        assert_eq!(
            simplify_options.shared.heuristic_poly,
            crate::HeuristicPoly::On
        );
        assert_eq!(
            simplify_options.shared.autoexpand_binomials,
            crate::AutoExpandBinomials::On
        );
        assert_eq!(simplify_options.budgets.max_total_rewrites, 123);
        assert_eq!(eval_options.steps_mode, crate::StepsMode::Compact);
        assert_eq!(eval_options.shared.heuristic_poly, crate::HeuristicPoly::On);
        assert_eq!(
            eval_options.shared.autoexpand_binomials,
            crate::AutoExpandBinomials::On
        );
        assert!(debug_mode);
    }
}
