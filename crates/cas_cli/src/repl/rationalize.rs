use super::*;

impl Repl {
    pub(crate) fn handle_rationalize(&mut self, line: &str) {
        let reply = self.handle_rationalize_core(line);
        self.print_reply(reply);
    }

    fn handle_rationalize_core(&mut self, line: &str) -> ReplReply {
        let rest = line.strip_prefix("rationalize").unwrap_or(line).trim();
        if rest.is_empty() {
            return reply_output(
                "Usage: rationalize <expr>\n\
                 Example: rationalize 1/(1 + sqrt(2) + sqrt(3))",
            );
        }

        match cas_solver::evaluate_rationalize_input(&mut self.core.engine.simplifier, rest) {
            Ok(out) => {
                // STYLE SNIFFING: Detect user's preferred notation BEFORE processing
                let user_style = cas_formatter::root_style::detect_root_style(
                    &self.core.engine.simplifier.context,
                    out.normalized_expr,
                );

                // Convert to string BEFORE mutable borrows to avoid borrow conflict
                let parsed_str = format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &self.core.engine.simplifier.context,
                        id: out.normalized_expr,
                    }
                );

                match out.outcome {
                    cas_solver::RationalizeEvalOutcome::Success { simplified_expr } => {
                        // Use DisplayExprStyled with detected style for consistent output
                        let style =
                            cas_formatter::root_style::StylePreferences::with_root_style(user_style);
                        let result_disp = cas_formatter::DisplayExprStyled::new(
                            &self.core.engine.simplifier.context,
                            simplified_expr,
                            &style,
                        );
                        reply_output(format!("Parsed: {}\nRationalized: {}", parsed_str, result_disp))
                    }
                    cas_solver::RationalizeEvalOutcome::NotApplicable => {
                        reply_output(format!(
                            "Parsed: {}\n\
                             Cannot rationalize: denominator is not a sum of surds\n\
                             (Supported: 1/(a + b√n + c√m) where a,b,c are rational and n,m are positive integers)",
                            parsed_str
                        ))
                    }
                    cas_solver::RationalizeEvalOutcome::BudgetExceeded => {
                        reply_output(format!(
                            "Parsed: {}\n\
                             Rationalization aborted: expression became too complex",
                            parsed_str
                        ))
                    }
                }
            }
            Err(cas_solver::RationalizeEvalError::Parse(e)) => {
                reply_output(format!("Parse error: {}", e))
            }
        }
    }
}
