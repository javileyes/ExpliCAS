use crate::eval_command_types::EvalResultLine;

pub(super) fn format_solution_set_result_line(
    context: &cas_ast::Context,
    solution_set: &cas_ast::SolutionSet,
) -> EvalResultLine {
    EvalResultLine {
        line: format!(
            "Result: {}",
            crate::display_solution_set(context, solution_set)
        ),
        terminal: false,
    }
}

pub(super) fn format_set_result_line() -> EvalResultLine {
    EvalResultLine {
        line: "Result: Set(...)".to_string(),
        terminal: false,
    }
}

pub(super) fn format_bool_result_line(value: bool) -> EvalResultLine {
    EvalResultLine {
        line: format!("Result: {}", value),
        terminal: false,
    }
}
