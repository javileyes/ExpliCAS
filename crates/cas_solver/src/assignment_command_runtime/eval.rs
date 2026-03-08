use crate::{
    apply_assignment_with_context, evaluate_assignment_command_with,
    evaluate_let_assignment_command_with, AssignmentApplyContext, AssignmentCommandOutput,
    Simplifier,
};

pub(super) fn evaluate_assignment_command_with_context<C: AssignmentApplyContext>(
    context: &mut C,
    simplifier: &mut Simplifier,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<AssignmentCommandOutput, String> {
    evaluate_assignment_command_with(name, expr_str, lazy, |name, expr_str, lazy| {
        apply_assignment_with_context(context, simplifier, name, expr_str, lazy)
    })
}

pub(super) fn evaluate_let_assignment_command_with_context<C: AssignmentApplyContext>(
    context: &mut C,
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<AssignmentCommandOutput, String> {
    evaluate_let_assignment_command_with(input, |name, expr_str, lazy| {
        apply_assignment_with_context(context, simplifier, name, expr_str, lazy)
    })
}
