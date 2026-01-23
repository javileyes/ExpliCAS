impl Repl {
    fn handle_rationalize(&mut self, line: &str) {
        use cas_engine::rationalize::{
            rationalize_denominator, RationalizeConfig, RationalizeResult,
        };

        let rest = line.strip_prefix("rationalize").unwrap_or(line).trim();
        if rest.is_empty() {
            println!("Usage: rationalize <expr>");
            println!("Example: rationalize 1/(1 + sqrt(2) + sqrt(3))");
            return;
        }

        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(parsed_expr) => {
                // CANONICALIZE: Rebuild tree to trigger Add auto-flatten at all levels
                // Parser creates tree incrementally, so nested Adds may not be flattened
                // normalize_core forces reconstruction ensuring canonical form
                let expr = cas_engine::canonical_forms::normalize_core(
                    &mut self.engine.simplifier.context,
                    parsed_expr,
                );
                // STYLE SNIFFING: Detect user's preferred notation BEFORE processing
                let user_style = cas_ast::detect_root_style(&self.engine.simplifier.context, expr);

                let disp = cas_ast::DisplayExpr {
                    context: &self.engine.simplifier.context,
                    id: expr,
                };
                println!("Parsed: {}", disp);

                let config = RationalizeConfig::default();
                let result =
                    rationalize_denominator(&mut self.engine.simplifier.context, expr, &config);

                match result {
                    RationalizeResult::Success(rationalized) => {
                        // Simplify the result
                        let (simplified, _) = self.engine.simplifier.simplify(rationalized);

                        // Use StyledExpr with detected style for consistent output
                        let result_disp = cas_ast::StyledExpr::new(
                            &self.engine.simplifier.context,
                            simplified,
                            user_style,
                        );
                        println!("Rationalized: {}", result_disp);
                    }
                    RationalizeResult::NotApplicable => {
                        println!("Cannot rationalize: denominator is not a sum of surds");
                        println!("(Supported: 1/(a + b√n + c√m) where a,b,c are rational and n,m are positive integers)");
                    }
                    RationalizeResult::BudgetExceeded => {
                        println!("Rationalization aborted: expression became too complex");
                    }
                }
            }
            Err(e) => println!("Parse error: {:?}", e),
        }
    }
}
