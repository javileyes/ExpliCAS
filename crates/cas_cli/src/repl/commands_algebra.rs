use super::*;
use crate::assumption_format;
use crate::result_format;

const TELESCOPE_USAGE_MESSAGE: &str = "Usage: telescope <expression>\n\
                 Example: telescope 1 + 2*cos(x) + 2*cos(2*x) - sin(5*x/2)/sin(x/2)";

const EXPAND_USAGE_MESSAGE: &str = "Usage: expand <expr>\n\
                 Description: Aggressively expands and distributes polynomials.\n\
                 Example: expand 1/2 * (sqrt(2) - 1) → sqrt(2)/2 - 1/2";

const EXPAND_LOG_USAGE_MESSAGE: &str = "Usage: expand_log <expr>\n\
                 Description: Expand logarithms using log properties.\n\
                 Transformations:\n\
                   ln(x*y)   → ln(x) + ln(y)\n\
                   ln(x/y)   → ln(x) - ln(y)\n\
                   ln(x^n)   → n * ln(x)\n\
                 Example: expand_log ln(x^2 * y) → 2*ln(x) + ln(y)";

fn extract_unary_command_tail<'a>(line: &'a str, command: &str) -> &'a str {
    line.strip_prefix(command).unwrap_or(line).trim()
}

fn parse_telescope_input(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("telescope").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest)
    }
}

fn parse_expand_input(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("expand").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest)
    }
}

fn parse_expand_log_input(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("expand_log").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest)
    }
}

fn wrap_expand_eval_expression(expr: &str) -> String {
    format!("expand({expr})")
}

fn format_telescope_eval_lines(input: &str, formatted_result: &str) -> Vec<String> {
    vec![format!("Parsed: {}\n\n{}", input, formatted_result)]
}

fn format_expand_log_eval_lines(
    context: &cas_ast::Context,
    parsed_expr: cas_ast::ExprId,
    expanded_expr: cas_ast::ExprId,
) -> Vec<String> {
    vec![
        format!(
            "Parsed: {}",
            cas_formatter::DisplayExpr {
                context,
                id: parsed_expr
            }
        ),
        format!(
            "Result: {}",
            cas_formatter::DisplayExpr {
                context,
                id: expanded_expr
            }
        ),
    ]
}

fn format_unary_function_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    result_expr: cas_ast::ExprId,
    steps: &[cas_solver::Step],
    function_name: &str,
    show_steps: bool,
    show_step_assumptions: bool,
) -> Vec<String> {
    let mut lines = vec![format!("Parsed: {}({})", function_name, input)];

    if show_steps && !steps.is_empty() {
        lines.push("Steps:".to_string());
        for (i, step) in steps.iter().enumerate() {
            lines.push(format!(
                "{}. {}  [{}]",
                i + 1,
                step.description,
                step.rule_name
            ));
            if show_step_assumptions {
                let assumption_events =
                    cas_solver::assumption_events_from_engine(step.assumption_events());
                for assumption_line in
                    assumption_format::format_displayable_assumption_lines(&assumption_events)
                {
                    lines.push(format!("   {}", assumption_line));
                }
            }
        }
    }

    lines.push(format!(
        "Result: {}",
        cas_formatter::DisplayExpr {
            context,
            id: result_expr
        }
    ));
    lines
}

fn evaluate_unary_command_lines(
    simplifier: &mut cas_solver::Simplifier,
    line: &str,
    command: &str,
    display_mode: SetDisplayMode,
    show_step_assumptions: bool,
    clean_result_line: bool,
) -> Result<Vec<String>, String> {
    let rest = extract_unary_command_tail(line, command);
    let parsed_expr = cas_parser::parse(rest, &mut simplifier.context)
        .map_err(|e| format!("Parse error: {e}"))?;
    let call_expr = simplifier.context.call(command, vec![parsed_expr]);
    let (result_expr, steps) = simplifier.simplify(call_expr);
    let mut lines = format_unary_function_eval_lines(
        &simplifier.context,
        rest,
        result_expr,
        &steps,
        command,
        display_mode != SetDisplayMode::None,
        show_step_assumptions,
    );
    if clean_result_line {
        result_format::clean_result_output_line(&mut lines);
    }
    Ok(lines)
}

impl Repl {
    pub(crate) fn handle_det_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        match evaluate_unary_command_lines(
            &mut self.core.engine.simplifier,
            line,
            "det",
            Self::set_display_mode_from_verbosity(verbosity),
            true,
            true,
        ) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_transpose_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        match evaluate_unary_command_lines(
            &mut self.core.engine.simplifier,
            line,
            "transpose",
            Self::set_display_mode_from_verbosity(verbosity),
            false,
            false,
        ) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_trace_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        match evaluate_unary_command_lines(
            &mut self.core.engine.simplifier,
            line,
            "trace",
            Self::set_display_mode_from_verbosity(verbosity),
            false,
            true,
        ) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    /// Handle the 'telescope' command for proving telescoping identities like Dirichlet kernel
    pub(crate) fn handle_telescope_core(&mut self, line: &str) -> ReplReply {
        let Some(rest) = parse_telescope_input(line) else {
            return reply_output(TELESCOPE_USAGE_MESSAGE);
        };
        let parsed_expr = match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => expr,
            Err(e) => return reply_output(format!("Parse error: {e}")),
        };
        let result = cas_solver::telescope(&mut self.core.engine.simplifier.context, parsed_expr);
        let formatted_result = result.format(&self.core.engine.simplifier.context);
        let lines = format_telescope_eval_lines(rest, &formatted_result);
        reply_output(lines.join("\n"))
    }

    /// Handle the 'expand' command for aggressive polynomial expansion
    /// Uses the engine `expand()` path which distributes without educational guards
    pub(crate) fn handle_expand_core(&mut self, line: &str) -> ReplReply {
        // Delegate to normal line processing with expand() function wrapper.
        // This ensures steps are shown, consistent with using expand() as a function.
        let wrapped = match parse_expand_input(line) {
            Some(rest) => wrap_expand_eval_expression(rest),
            None => return reply_output(EXPAND_USAGE_MESSAGE),
        };
        self.handle_eval_core(&wrapped)
    }

    /// Handle the 'expand_log' command for explicit logarithm expansion
    /// Expands ln(xy) → ln(x) + ln(y), ln(x/y) → ln(x) - ln(y), ln(x^n) → n*ln(x)
    pub(crate) fn handle_expand_log_core(&mut self, line: &str) -> ReplReply {
        let Some(rest) = parse_expand_log_input(line) else {
            return reply_output(EXPAND_LOG_USAGE_MESSAGE);
        };
        let parsed_expr = match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => expr,
            Err(e) => return reply_output(format!("Parse error: {e}")),
        };
        let expanded_expr =
            cas_solver::expand_log_recursive(&mut self.core.engine.simplifier.context, parsed_expr);
        let mut lines = format_expand_log_eval_lines(
            &self.core.engine.simplifier.context,
            parsed_expr,
            expanded_expr,
        );
        result_format::clean_result_output_line(&mut lines);
        reply_output(lines.join("\n"))
    }
}
