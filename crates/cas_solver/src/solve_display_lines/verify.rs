pub(super) fn push_solution_verification_lines(
    lines: &mut Vec<String>,
    simplifier: &mut crate::Simplifier,
    original_equation: Option<&cas_ast::Equation>,
    output: &crate::EvalOutputView,
    config: crate::SolveCommandRenderConfig,
    var: &str,
) {
    if !config.check_solutions {
        return;
    }

    if let crate::EvalResult::SolutionSet(solution_set) = &output.result {
        if let Some(eq) = original_equation {
            let verify_result = crate::api::verify_solution_set(simplifier, eq, var, solution_set);
            lines.extend(crate::format_verify_summary_lines(
                &simplifier.context,
                var,
                &verify_result,
                "  ",
            ));
        }
    }
}
