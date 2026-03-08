use crate::SemanticsCommandOutput;

pub(super) fn show_semantics_output(
    simplify_options: &mut crate::SimplifyOptions,
    eval_options: &mut crate::EvalOptions,
) -> SemanticsCommandOutput {
    let state = crate::semantics_view_state_from_options(simplify_options, eval_options);
    SemanticsCommandOutput {
        lines: crate::format_semantics_overview_lines(&state),
        sync_simplifier: false,
    }
}

pub(super) fn help_semantics_output() -> SemanticsCommandOutput {
    SemanticsCommandOutput {
        lines: vec![crate::semantics_help_message().to_string()],
        sync_simplifier: false,
    }
}

pub(super) fn axis_semantics_output(
    simplify_options: &mut crate::SimplifyOptions,
    eval_options: &mut crate::EvalOptions,
    axis: &str,
) -> SemanticsCommandOutput {
    let state = crate::semantics_view_state_from_options(simplify_options, eval_options);
    SemanticsCommandOutput {
        lines: crate::format_semantics_axis_lines(&state, axis),
        sync_simplifier: false,
    }
}

pub(super) fn unknown_semantics_output(subcommand: &str) -> SemanticsCommandOutput {
    SemanticsCommandOutput {
        lines: vec![crate::format_semantics_unknown_subcommand_message(
            subcommand,
        )],
        sync_simplifier: false,
    }
}
