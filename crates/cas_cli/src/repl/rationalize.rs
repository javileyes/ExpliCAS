use super::*;

impl Repl {
    pub(crate) fn handle_rationalize(&mut self, line: &str) {
        let reply = self.handle_rationalize_core(line);
        self.print_reply(reply);
    }

    fn handle_rationalize_core(&mut self, line: &str) -> ReplReply {
        use cas_engine::rationalize::{
            rationalize_denominator, RationalizeConfig, RationalizeResult,
        };

        let rest = line.strip_prefix("rationalize").unwrap_or(line).trim();
        if rest.is_empty() {
            return reply_output(
                "Usage: rationalize <expr>\n\
                 Example: rationalize 1/(1 + sqrt(2) + sqrt(3))",
            );
        }

        match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(parsed_expr) => {
                // CANONICALIZE: Rebuild tree to trigger Add auto-flatten at all levels
                // Parser creates tree incrementally, so nested Adds may not be flattened
                // normalize_core forces reconstruction ensuring canonical form
                let expr = cas_engine::canonical_forms::normalize_core(
                    &mut self.core.engine.simplifier.context,
                    parsed_expr,
                );
                // STYLE SNIFFING: Detect user's preferred notation BEFORE processing
                let user_style =
                    cas_ast::detect_root_style(&self.core.engine.simplifier.context, expr);

                // Convert to string BEFORE mutable borrows to avoid borrow conflict
                let parsed_str = format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &self.core.engine.simplifier.context,
                        id: expr,
                    }
                );

                let config = RationalizeConfig::default();
                let result = rationalize_denominator(
                    &mut self.core.engine.simplifier.context,
                    expr,
                    &config,
                );

                match result {
                    RationalizeResult::Success(rationalized) => {
                        // Simplify the result
                        let (simplified, _) = self.core.engine.simplifier.simplify(rationalized);

                        // Use DisplayExprStyled with detected style for consistent output
                        let style = cas_ast::StylePreferences::with_root_style(user_style);
                        let result_disp = cas_formatter::DisplayExprStyled::new(
                            &self.core.engine.simplifier.context,
                            simplified,
                            &style,
                        );
                        reply_output(format!("Parsed: {}\nRationalized: {}", parsed_str, result_disp))
                    }
                    RationalizeResult::NotApplicable => {
                        reply_output(format!(
                            "Parsed: {}\n\
                             Cannot rationalize: denominator is not a sum of surds\n\
                             (Supported: 1/(a + b√n + c√m) where a,b,c are rational and n,m are positive integers)",
                            parsed_str
                        ))
                    }
                    RationalizeResult::BudgetExceeded => {
                        reply_output(format!(
                            "Parsed: {}\n\
                             Rationalization aborted: expression became too complex",
                            parsed_str
                        ))
                    }
                }
            }
            Err(e) => reply_output(format!("Parse error: {:?}", e)),
        }
    }
}
