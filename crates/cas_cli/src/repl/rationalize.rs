use super::*;

const RATIONALIZE_USAGE_MESSAGE: &str = "Usage: rationalize <expr>\n\
                 Example: rationalize 1/(1 + sqrt(2) + sqrt(3))";

fn parse_rationalize_input(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("rationalize").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RationalizeCliOutcome {
    Success(cas_ast::ExprId),
    NotApplicable,
    BudgetExceeded,
}

fn evaluate_rationalize_input(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
) -> Result<(cas_ast::ExprId, RationalizeCliOutcome), String> {
    let parsed_expr =
        cas_parser::parse(input, &mut simplifier.context).map_err(|e| format!("{:?}", e))?;
    let normalized_expr =
        cas_solver::canonical_forms::normalize_core(&mut simplifier.context, parsed_expr);
    let config = cas_solver::rationalize::RationalizeConfig::default();
    let rationalized = cas_solver::rationalize::rationalize_denominator(
        &mut simplifier.context,
        normalized_expr,
        &config,
    );
    let outcome = match rationalized {
        cas_solver::rationalize::RationalizeResult::Success(expr) => {
            RationalizeCliOutcome::Success(simplifier.simplify(expr).0)
        }
        cas_solver::rationalize::RationalizeResult::NotApplicable => {
            RationalizeCliOutcome::NotApplicable
        }
        cas_solver::rationalize::RationalizeResult::BudgetExceeded => {
            RationalizeCliOutcome::BudgetExceeded
        }
    };
    Ok((normalized_expr, outcome))
}

fn format_rationalize_eval_lines(
    context: &cas_ast::Context,
    normalized_expr: cas_ast::ExprId,
    outcome: RationalizeCliOutcome,
) -> Vec<String> {
    let user_style = cas_formatter::root_style::detect_root_style(context, normalized_expr);
    let parsed = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context,
            id: normalized_expr
        }
    );

    let line = match outcome {
        RationalizeCliOutcome::Success(simplified_expr) => {
            let style = cas_formatter::root_style::StylePreferences::with_root_style(user_style);
            let rendered = cas_formatter::DisplayExprStyled::new(context, simplified_expr, &style);
            format!("Parsed: {}\nRationalized: {}", parsed, rendered)
        }
        RationalizeCliOutcome::NotApplicable => format!(
            "Parsed: {}\n\
             Cannot rationalize: denominator is not a sum of surds\n\
             (Supported: 1/(a + b√n + c√m) where a,b,c are rational and n,m are positive integers)",
            parsed
        ),
        RationalizeCliOutcome::BudgetExceeded => format!(
            "Parsed: {}\n\
             Rationalization aborted: expression became too complex",
            parsed
        ),
    };

    vec![line]
}

impl Repl {
    pub(crate) fn handle_rationalize_core(&mut self, line: &str) -> ReplReply {
        let Some(rest) = parse_rationalize_input(line) else {
            return reply_output(RATIONALIZE_USAGE_MESSAGE);
        };

        let (normalized_expr, outcome) =
            match evaluate_rationalize_input(&mut self.core.engine.simplifier, rest) {
                Ok(output) => output,
                Err(error) => return reply_output(format!("Parse error: {}", error)),
            };

        let lines = format_rationalize_eval_lines(
            &self.core.engine.simplifier.context,
            normalized_expr,
            outcome,
        );
        reply_output(lines.join("\n"))
    }
}
