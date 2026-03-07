use cas_ast::Context;

pub(super) fn push_verbose_substep_lines(
    lines: &mut Vec<String>,
    ctx: &Context,
    step_index: usize,
    substeps: &[crate::SolveSubStep],
) {
    for (substep_index, substep) in substeps.iter().enumerate() {
        let sub_lhs = cas_formatter::DisplayExpr {
            context: ctx,
            id: substep.equation_after.lhs,
        }
        .to_string();
        let sub_rhs = cas_formatter::DisplayExpr {
            context: ctx,
            id: substep.equation_after.rhs,
        }
        .to_string();
        lines.push(format!(
            "      {}.{}. {}",
            step_index + 1,
            substep_index + 1,
            substep.description
        ));
        lines.push(format!(
            "          -> {} {} {}",
            sub_lhs, substep.equation_after.op, sub_rhs
        ));
    }
}
