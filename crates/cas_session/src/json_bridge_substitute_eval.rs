use crate::json_bridge_substitute_parse::parse_substitute_input;
use crate::json_bridge_substitute_types::SubstituteParseIssue;
use cas_api_models::{SubstituteEvalResult, SubstituteEvalStep};
use cas_ast::hold::strip_all_holds;

pub(crate) fn eval_substitute_impl(
    expr_str: &str,
    target_str: &str,
    replacement_str: &str,
    mode: &str,
    steps_enabled: bool,
) -> Result<SubstituteEvalResult, SubstituteParseIssue> {
    let (mut ctx, expr, target, replacement) =
        parse_substitute_input(expr_str, target_str, replacement_str)?;

    let sub_opts = match mode {
        "exact" => cas_solver::substitute::SubstituteOptions::exact(),
        _ => cas_solver::substitute::SubstituteOptions::power_aware_no_remainder(),
    };

    let sub_result = cas_solver::substitute::substitute_with_steps(
        &mut ctx,
        expr,
        target,
        replacement,
        sub_opts,
    );

    let clean_result = strip_all_holds(&mut ctx, sub_result.expr);
    let result = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: clean_result
        }
    );

    let steps = if steps_enabled {
        sub_result
            .steps
            .into_iter()
            .map(|step| SubstituteEvalStep {
                rule: step.rule,
                before: step.before,
                after: step.after,
                note: step.note,
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(SubstituteEvalResult { result, steps })
}
