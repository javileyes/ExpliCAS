use crate::SemanticsCommandOutput;

pub(super) fn set_semantics_output(
    args: Vec<String>,
    simplify_options: &mut crate::SimplifyOptions,
    eval_options: &mut crate::EvalOptions,
) -> SemanticsCommandOutput {
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    match crate::evaluate_semantics_set_args_to_overview_lines(
        &refs,
        simplify_options,
        eval_options,
    ) {
        Ok(lines) => SemanticsCommandOutput {
            lines,
            sync_simplifier: true,
        },
        Err(error) => SemanticsCommandOutput {
            lines: vec![error],
            sync_simplifier: false,
        },
    }
}

pub(super) fn preset_semantics_output(
    args: Vec<String>,
    simplify_options: &mut crate::SimplifyOptions,
    eval_options: &mut crate::EvalOptions,
) -> SemanticsCommandOutput {
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let out =
        crate::evaluate_semantics_preset_args_to_options(&refs, simplify_options, eval_options);
    SemanticsCommandOutput {
        lines: out.lines,
        sync_simplifier: out.applied,
    }
}
