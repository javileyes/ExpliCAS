use cas_api_models::{SubstituteEvalResult, SubstituteEvalStep};
use cas_ast::hold::strip_all_holds;

use super::parse::{parse_substitute_input, SubstituteParseIssue};

fn substitute_options_for_mode(mode: &str) -> crate::substitute::SubstituteOptions {
    match mode {
        "exact" => crate::substitute::SubstituteOptions::exact(),
        _ => crate::substitute::SubstituteOptions::power_aware_no_remainder(),
    }
}

pub fn eval_substitute_impl(
    expr_str: &str,
    target_str: &str,
    with_str: &str,
    mode: &str,
    steps_enabled: bool,
) -> Result<SubstituteEvalResult, SubstituteParseIssue> {
    let (mut ctx, expr, target, replacement) =
        parse_substitute_input(expr_str, target_str, with_str)?;

    let sub_result = crate::substitute::substitute_with_steps(
        &mut ctx,
        expr,
        target,
        replacement,
        substitute_options_for_mode(mode),
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
            .map(|s| SubstituteEvalStep {
                rule: s.rule,
                before: s.before,
                after: s.after,
                note: s.note,
            })
            .collect()
    } else {
        vec![]
    };

    Ok(SubstituteEvalResult { result, steps })
}
