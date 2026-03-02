use super::*;

const LIMIT_USAGE_MESSAGE: &str = "Usage: limit <expr> [, <var> [, <direction> [, safe]]]\n\
                 Examples:\n\
                   limit x^2                      → infinity (default: x → +∞)\n\
                   limit (x^2+1)/(2*x^2-3), x     → 1/2\n\
                   limit x^3/x^2, x, -infinity    → -infinity\n\
                   limit (x-x)/x, x, infinity, safe → 0 (with pre-simplify)";

fn extract_limit_command_tail(line: &str) -> &str {
    line.strip_prefix("limit").unwrap_or(line).trim()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LimitCommandInput<'a> {
    expr: &'a str,
    var: &'a str,
    approach: cas_solver::Approach,
    presimplify: cas_solver::PreSimplifyMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LimitCommandEvalError {
    EmptyInput,
    Parse(String),
    Limit(String),
}

#[derive(Debug, Clone)]
struct LimitCommandEvalOutput {
    var: String,
    approach: cas_solver::Approach,
    result: String,
    warning: Option<String>,
}

fn split_by_comma_ignoring_parens(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut balance = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' | '{' => balance += 1,
            ')' | ']' | '}' => balance -= 1,
            ',' if balance == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    if start < s.len() {
        parts.push(&s[start..]);
    }

    parts
}

fn parse_limit_command_input(rest: &str) -> LimitCommandInput<'_> {
    let parts = split_by_comma_ignoring_parens(rest);
    let expr = parts.first().copied().unwrap_or("").trim();
    let var = parts.get(1).copied().unwrap_or("x").trim();
    let dir = parts.get(2).copied().unwrap_or("infinity").trim();
    let mode = parts.get(3).copied().unwrap_or("off").trim();

    let approach = if dir.contains("-infinity") || dir.contains("-inf") {
        cas_solver::Approach::NegInfinity
    } else {
        cas_solver::Approach::PosInfinity
    };
    let presimplify = if mode.eq_ignore_ascii_case("safe") {
        cas_solver::PreSimplifyMode::Safe
    } else {
        cas_solver::PreSimplifyMode::Off
    };

    LimitCommandInput {
        expr,
        var,
        approach,
        presimplify,
    }
}

fn evaluate_limit_command_input(
    input: &str,
) -> Result<LimitCommandEvalOutput, LimitCommandEvalError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(LimitCommandEvalError::EmptyInput);
    }

    let parsed = parse_limit_command_input(trimmed);
    match cas_solver::json::eval_limit_from_str(
        parsed.expr,
        parsed.var,
        parsed.approach,
        parsed.presimplify,
    ) {
        Ok(limit_result) => Ok(LimitCommandEvalOutput {
            var: parsed.var.to_string(),
            approach: parsed.approach,
            result: limit_result.result,
            warning: limit_result.warning,
        }),
        Err(cas_solver::json::LimitEvalError::Parse(message)) => {
            Err(LimitCommandEvalError::Parse(message))
        }
        Err(cas_solver::json::LimitEvalError::Limit(message)) => {
            Err(LimitCommandEvalError::Limit(message))
        }
    }
}

fn format_limit_command_error_message(error: &LimitCommandEvalError) -> String {
    match error {
        LimitCommandEvalError::EmptyInput => LIMIT_USAGE_MESSAGE.to_string(),
        LimitCommandEvalError::Parse(message) => message.clone(),
        LimitCommandEvalError::Limit(message) => {
            format!("Error computing limit: {}", message)
        }
    }
}

fn format_limit_command_eval_lines(output: &LimitCommandEvalOutput) -> Vec<String> {
    let dir_disp = match output.approach {
        cas_solver::Approach::PosInfinity => "+∞",
        cas_solver::Approach::NegInfinity => "-∞",
    };
    let mut lines = vec![format!(
        "lim_{{{}→{}}} = {}",
        output.var, dir_disp, output.result
    )];
    if let Some(warning) = &output.warning {
        lines.push(format!("Warning: {}", warning));
    }
    lines
}

impl Repl {
    pub(crate) fn handle_limit_core(&mut self, line: &str) -> ReplReply {
        let rest = extract_limit_command_tail(line);
        if rest.is_empty() {
            return reply_output(LIMIT_USAGE_MESSAGE);
        }

        let output = match evaluate_limit_command_input(rest) {
            Ok(output) => output,
            Err(error) => return reply_output(format_limit_command_error_message(&error)),
        };

        let lines = format_limit_command_eval_lines(&output);
        reply_output(lines.join("\n"))
    }
}
