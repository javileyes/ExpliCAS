use cas_ast::{Expr, ExprId};
use cas_formatter::DisplayExpr;

pub(crate) struct EquivCommandOutput {
    pub(crate) result: crate::EquivalenceResult,
    pub(crate) residual: Option<String>,
}

pub(crate) fn simplified_equivalence_residual_expr(
    simplifier: &mut crate::Simplifier,
    lhs: ExprId,
    rhs: ExprId,
) -> ExprId {
    let residual = simplifier.context.add(Expr::Sub(lhs, rhs));
    let previous_steps_mode = simplifier.steps_mode;
    simplifier.steps_mode = cas_engine::StepsMode::Off;
    let (simplified, _) = simplifier.simplify(residual);
    simplifier.steps_mode = previous_steps_mode;
    simplified
}

fn residual_for_false_equivalence(
    simplifier: &mut crate::Simplifier,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
    result: &crate::EquivalenceResult,
) -> Option<String> {
    if !matches!(result, crate::EquivalenceResult::False) {
        return None;
    }

    let residual = simplified_equivalence_residual_expr(simplifier, lhs, rhs);
    Some(
        DisplayExpr {
            context: &simplifier.context,
            id: residual,
        }
        .to_string(),
    )
}

fn evaluate_equiv_input(
    simplifier: &mut crate::Simplifier,
    input: &str,
) -> Result<EquivCommandOutput, crate::ParseExprPairError> {
    let (lhs, rhs) = crate::parse_expr_pair(&mut simplifier.context, input)?;
    let result = simplifier.are_equivalent_extended(lhs, rhs);
    let residual = residual_for_false_equivalence(simplifier, lhs, rhs, &result);
    Ok(EquivCommandOutput { result, residual })
}

/// Evaluate equivalence command input and format user-facing output lines.
pub(crate) fn evaluate_equiv_command_lines(
    simplifier: &mut crate::Simplifier,
    input: &str,
) -> Result<Vec<String>, crate::ParseExprPairError> {
    let output = evaluate_equiv_input(simplifier, input)?;
    Ok(crate::format_equiv_command_output_lines(&output))
}

/// Evaluate `equiv` command input and return user-facing message text.
pub(crate) fn evaluate_equiv_command_message(
    simplifier: &mut crate::Simplifier,
    input: &str,
) -> Result<String, crate::ParseExprPairError> {
    Ok(evaluate_equiv_command_lines(simplifier, input)?.join("\n"))
}

/// Evaluate full `equiv ...` invocation and return user-facing message text.
pub(crate) fn evaluate_equiv_invocation_message(
    simplifier: &mut crate::Simplifier,
    line: &str,
) -> Result<String, String> {
    let input = crate::extract_equiv_command_tail(line);
    evaluate_equiv_command_message(simplifier, input)
        .map_err(|error| crate::format_expr_pair_parse_error_message(&error, "equiv"))
}
