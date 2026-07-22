use cas_api_models::{parse_eval_special_command, EvalLimitApproach, EvalSpecialCommand};
use cas_ast::{Context, ExprId};
use cas_formatter::{latex_escape, ParseStyleSignals};

fn style_latex_for_input(ctx: &Context, id: ExprId, signals: &ParseStyleSignals) -> String {
    crate::eval_output_latex_style::render_expr_latex_for_eval(
        ctx,
        id,
        signals,
        crate::eval_output_latex_style::EvalLatexRenderIntent::Input,
    )
}

/// LaTeX for a dsolve INPUT equation: inside an ODE command the unknown is a
/// DEPENDENT function of the single independent variable, so every derivative
/// is ORDINARY — `d/dx`, never `∂/∂x` (D14, Fase 4). The general
/// `diff_is_partial` heuristic (multi-variable ⇒ ∂) is right everywhere else;
/// this rewrite is scoped to the dsolve input channel only.
fn ode_input_latex(ctx: &Context, id: ExprId, signals: &ParseStyleSignals) -> String {
    style_latex_for_input(ctx, id, signals).replace("\\partial", "d")
}

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

fn fallback_solve_input_latex(equation: &str, var: &str) -> String {
    format!(
        "\\operatorname{{solve}}\\left(\\texttt{{{}}}, {}\\right)",
        latex_escape(equation),
        latex_escape(var)
    )
}

/// LaTeX for a relational operator, so an INEQUALITY argument of `solve(...)` echoes its real relation
/// (`>`, `≤`, …) instead of collapsing to `=`.
fn relop_latex(op: &cas_ast::RelOp) -> &'static str {
    use cas_ast::RelOp;
    match op {
        RelOp::Eq => "=",
        RelOp::Neq => "\\neq",
        RelOp::Lt => "<",
        RelOp::Gt => ">",
        RelOp::Leq => "\\leq",
        RelOp::Geq => "\\geq",
    }
}

fn format_solve_input_latex(equation: &str, var: &str) -> String {
    let mut temp_ctx = cas_ast::Context::new();
    let statement = match cas_parser::parse_statement(equation, &mut temp_ctx) {
        Ok(statement) => statement,
        Err(_) => return fallback_solve_input_latex(equation, var),
    };
    let cas_parser::Statement::Equation(eq) = statement else {
        return fallback_solve_input_latex(equation, var);
    };

    let eq_signals = ParseStyleSignals::from_input_string(equation);
    let lhs = style_latex_for_input(&temp_ctx, eq.lhs, &eq_signals);
    let rhs = style_latex_for_input(&temp_ctx, eq.rhs, &eq_signals);
    let op = relop_latex(&eq.op);

    format!(
        "\\operatorname{{solve}}\\left({lhs} {op} {rhs}, {}\\right)",
        latex_escape(var)
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
        let eq_signals = ParseStyleSignals::from_input_string(eq_str);
        let lhs = style_latex_for_input(&temp_ctx, eq.lhs, &eq_signals);
        let rhs = style_latex_for_input(&temp_ctx, eq.rhs, &eq_signals);
        let op = relop_latex(&eq.op);
        rendered_parts.push(format!("{lhs} {op} {rhs}"));
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
    equiv_target: Option<ExprId>,
    signals: &ParseStyleSignals,
) -> String {
    if let Some(command) = parse_eval_special_command(raw_input) {
        match command {
            EvalSpecialCommand::Limit { var, approach, .. } => {
                let expr_latex = style_latex_for_input(ctx, parsed, signals);
                let approach_latex = match approach {
                    EvalLimitApproach::PosInfinity => "\\infty".to_string(),
                    EvalLimitApproach::NegInfinity => "-\\infty".to_string(),
                    EvalLimitApproach::Finite(point) => latex_escape(&point),
                    EvalLimitApproach::FiniteFromLeft(point) => {
                        format!("{}^-", latex_escape(&point))
                    }
                    EvalLimitApproach::FiniteFromRight(point) => {
                        format!("{}^+", latex_escape(&point))
                    }
                };
                return format!("\\lim_{{{var} \\to {approach_latex}}} {expr_latex}");
            }
            EvalSpecialCommand::Derive { .. } => {
                if let Some(target) = derive_target {
                    let parsed_latex = style_latex_for_input(ctx, parsed, signals);
                    let target_latex = style_latex_for_input(ctx, target, signals);
                    return format!(
                        "\\operatorname{{derive}}\\left({parsed_latex}, {target_latex}\\right)"
                    );
                }
            }
            EvalSpecialCommand::Equiv { .. } => {
                if let Some(target) = equiv_target {
                    let parsed_latex = style_latex_for_input(ctx, parsed, signals);
                    let target_latex = style_latex_for_input(ctx, target, signals);
                    return format!("{parsed_latex} \\leftrightarrow {target_latex}");
                }
            }
            EvalSpecialCommand::SolveSystem { input } => {
                return format_solve_system_input_latex(&input);
            }
            EvalSpecialCommand::Solve { equation, var } => {
                return format_solve_input_latex(&equation, &var);
            }
            EvalSpecialCommand::DsolveSystem { funcs, var, .. } => {
                let eq_latex = ode_input_latex(ctx, parsed, signals);
                let f_list = funcs.join(", ");
                return format!(
                    "\\operatorname{{dsolve}}\\left(\\left[{eq_latex}, \\ldots\\right],\\; \\left[{f_list}\\right],\\; {var}\\right)"
                );
            }
            EvalSpecialCommand::Dsolve { func, var, .. } => {
                // The parsed tree IS the ODE equation (`Equal(lhs, rhs)`); render
                // it as the equation with the unknown/variable named.
                let eq_latex = if let cas_ast::Expr::Function(fn_id, args) = ctx.get(parsed) {
                    if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Equal) && args.len() == 2 {
                        let lhs_latex = ode_input_latex(ctx, args[0], signals);
                        let rhs_latex = ode_input_latex(ctx, args[1], signals);
                        format!("{lhs_latex} = {rhs_latex}")
                    } else {
                        ode_input_latex(ctx, parsed, signals)
                    }
                } else {
                    ode_input_latex(ctx, parsed, signals)
                };
                return format!(
                    "\\operatorname{{dsolve}}\\left({eq_latex},\\; {func},\\; {var}\\right)"
                );
            }
        }
    }

    if let Some((lhs, rhs)) = cas_ast::eq::unwrap_eq(ctx, parsed) {
        let lhs_latex = style_latex_for_input(ctx, lhs, signals);
        let rhs_latex = style_latex_for_input(ctx, rhs, signals);
        format!("{lhs_latex} = {rhs_latex}")
    } else {
        style_latex_for_input(ctx, parsed, signals)
    }
}
