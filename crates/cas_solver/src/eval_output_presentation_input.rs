use cas_api_models::{parse_eval_special_command, EvalLimitApproach, EvalSpecialCommand};
use cas_ast::{Context, ExprId};
use cas_formatter::{latex_escape, LaTeXExpr};

fn split_solve_system_parts(input: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0_i32;
    let mut start = 0;

    for (i, ch) in input.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = (depth - 1).max(0),
            ';' if depth == 0 => {
                parts.push(input[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(input[start..].trim());
    parts
}

fn fallback_solve_system_input_latex(input: &str) -> String {
    format!(
        "\\operatorname{{solve\\_system}}\\left(\\texttt{{{}}}\\right)",
        latex_escape(input)
    )
}

fn format_solve_system_input_latex(input: &str) -> String {
    let parts = split_solve_system_parts(input);
    if parts.len() < 4 || !parts.len().is_multiple_of(2) {
        return fallback_solve_system_input_latex(input);
    }

    let n = parts.len() / 2;
    let eq_parts = &parts[..n];
    let var_parts = &parts[n..];
    let mut temp_ctx = cas_ast::Context::new();
    let mut rendered_parts = Vec::with_capacity(parts.len());

    for eq_str in eq_parts {
        let statement = match cas_parser::parse_statement(eq_str, &mut temp_ctx) {
            Ok(statement) => statement,
            Err(_) => return fallback_solve_system_input_latex(input),
        };
        let cas_parser::Statement::Equation(eq) = statement else {
            return fallback_solve_system_input_latex(input);
        };
        let lhs = LaTeXExpr {
            context: &temp_ctx,
            id: eq.lhs,
        }
        .to_latex();
        let rhs = LaTeXExpr {
            context: &temp_ctx,
            id: eq.rhs,
        }
        .to_latex();
        rendered_parts.push(format!("{lhs} = {rhs}"));
    }

    rendered_parts.extend(var_parts.iter().map(|var| latex_escape(var)));
    format!(
        "\\operatorname{{solve\\_system}}\\left({}\\right)",
        rendered_parts.join(";\\ ")
    )
}

pub(crate) fn format_output_input_latex(
    ctx: &Context,
    raw_input: &str,
    parsed: ExprId,
    derive_target: Option<ExprId>,
) -> String {
    if let Some(command) = parse_eval_special_command(raw_input) {
        match command {
            EvalSpecialCommand::Limit { var, approach, .. } => {
                let expr_latex = LaTeXExpr {
                    context: ctx,
                    id: parsed,
                }
                .to_latex();
                let approach_latex = match approach {
                    EvalLimitApproach::PosInfinity => "\\infty",
                    EvalLimitApproach::NegInfinity => "-\\infty",
                };
                return format!("\\lim_{{{var} \\to {approach_latex}}} {expr_latex}");
            }
            EvalSpecialCommand::Derive { .. } => {
                if let Some(target) = derive_target {
                    let parsed_latex = LaTeXExpr {
                        context: ctx,
                        id: parsed,
                    }
                    .to_latex();
                    let target_latex = LaTeXExpr {
                        context: ctx,
                        id: target,
                    }
                    .to_latex();
                    return format!(
                        "\\operatorname{{derive}}\\left({parsed_latex}, {target_latex}\\right)"
                    );
                }
            }
            EvalSpecialCommand::SolveSystem { input } => {
                return format_solve_system_input_latex(&input);
            }
            EvalSpecialCommand::Solve { .. } => {}
        }
    }

    if let Some((lhs, rhs)) = cas_ast::eq::unwrap_eq(ctx, parsed) {
        let lhs_latex = LaTeXExpr {
            context: ctx,
            id: lhs,
        }
        .to_latex();
        let rhs_latex = LaTeXExpr {
            context: ctx,
            id: rhs,
        }
        .to_latex();
        format!("{lhs_latex} = {rhs_latex}")
    } else {
        LaTeXExpr {
            context: ctx,
            id: parsed,
        }
        .to_latex()
    }
}
