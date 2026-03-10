use cas_api_models::EvalJsonOutput;
use cas_ast::ExprId;

use crate::eval_output_finalize_expr::finalize_expr_like_eval_output;
use crate::eval_output_finalize_input::{EvalOutputFinalizeContext, EvalOutputFinalizeInput};
use crate::eval_output_finalize_nonexpr::{finalize_bool_output, finalize_solution_set_output};

fn expr_like_result_id(result: &crate::EvalResult) -> Option<ExprId> {
    match result {
        crate::EvalResult::Expr(e) => Some(*e),
        crate::EvalResult::Set(v) if !v.is_empty() => Some(v[0]),
        _ => None,
    }
}

pub(crate) fn finalize_eval_output(
    input: EvalOutputFinalizeInput<'_>,
) -> Result<EvalJsonOutput, String> {
    let (
        EvalOutputFinalizeContext {
            result,
            ctx,
            max_chars,
        },
        shared,
    ) = input.split();

    if let Some(result_expr) = expr_like_result_id(result) {
        return Ok(finalize_expr_like_eval_output(
            ctx,
            result_expr,
            max_chars,
            shared,
        ));
    }

    match result {
        crate::EvalResult::SolutionSet(solution_set) => {
            Ok(finalize_solution_set_output(ctx, solution_set, shared))
        }
        crate::EvalResult::Bool(b) => Ok(finalize_bool_output(*b, shared)),
        _ => Err("No result expression".to_string()),
    }
}
