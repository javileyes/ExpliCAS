use super::*;

impl Repl {
    pub(crate) fn handle_equiv(&mut self, line: &str) {
        let reply = self.handle_equiv_core(line);
        self.print_reply(reply);
    }

    fn handle_equiv_core(&mut self, line: &str) -> ReplReply {
        use cas_parser::Statement;

        let rest = line[6..].trim();
        if let Some((expr1_str, expr2_str)) = rsplit_ignoring_parens(rest, ',') {
            let expr1_str = expr1_str.trim();
            let expr2_str = expr2_str.trim();

            // Helper to parse string to ExprId
            fn parse_arg(s: &str, ctx: &mut cas_ast::Context) -> Result<cas_ast::ExprId, String> {
                if s.starts_with('#') && s[1..].chars().all(char::is_numeric) {
                    Ok(ctx.var(s))
                } else {
                    match cas_parser::parse_statement(s, ctx) {
                        Ok(Statement::Equation(eq)) => Ok(ctx.call("Equal", vec![eq.lhs, eq.rhs])),
                        Ok(Statement::Expression(e)) => Ok(e),
                        Err(e) => Err(format!("{}", e)),
                    }
                }
            }

            let e1_res = parse_arg(expr1_str, &mut self.core.engine.simplifier.context);
            let e2_res = parse_arg(expr2_str, &mut self.core.engine.simplifier.context);

            match (e1_res, e2_res) {
                (Ok(e1), Ok(e2)) => {
                    // V2.14.45: Use new tri-state equivalence check
                    use cas_engine::EquivalenceResult;

                    let result = self.core.engine.simplifier.are_equivalent_extended(e1, e2);

                    let mut lines = Vec::new();
                    match result {
                        EquivalenceResult::True => {
                            lines.push("True".to_string());
                        }
                        EquivalenceResult::ConditionalTrue { requires } => {
                            lines.push("True (conditional)".to_string());
                            if !requires.is_empty() {
                                lines.push("ℹ️ Requires:".to_string());
                                for req in &requires {
                                    lines.push(format!("  • {}", req));
                                }
                            }
                        }
                        EquivalenceResult::False => {
                            lines.push("False".to_string());
                        }
                        EquivalenceResult::Unknown => {
                            lines.push("Unknown (cannot prove equivalence)".to_string());
                        }
                    }
                    reply_output(lines.join("\n"))
                }
                (Err(e), _) => reply_output(format!("Error parsing first arg: {}", e)),
                (_, Err(e)) => reply_output(format!("Error parsing second arg: {}", e)),
            }
        } else {
            reply_output("Usage: equiv <expr1>, <expr2>")
        }
    }

    pub(crate) fn handle_subst(&mut self, line: &str) {
        let reply = self.handle_subst_core(line, self.verbosity);
        self.print_reply(reply);
    }

    fn handle_subst_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        // Format: subst <expr>, <target>, <replacement>
        // Examples:
        //   subst x^4 + x^2 + 1, x^2, y   → y² + y + 1 (power-aware)
        //   subst x^2 + x, x, 3          → 12 (variable substitution)
        let rest = line[6..].trim();

        // Split by commas (respecting parentheses)
        let parts: Vec<&str> = split_by_comma_ignoring_parens(rest);

        if parts.len() != 3 {
            return reply_output(
                "Usage: subst <expr>, <target>, <replacement>\n\n\
                 Examples:\n\
                   subst x^2 + x, x, 3              → 12\n\
                   subst x^4 + x^2 + 1, x^2, y      → y² + y + 1\n\
                   subst x^3, x^2, y                → y·x",
            );
        }

        let expr_str = parts[0].trim();
        let target_str = parts[1].trim();
        let replacement_str = parts[2].trim();

        // Parse the main expression
        let expr = match cas_parser::parse(expr_str, &mut self.core.engine.simplifier.context) {
            Ok(e) => e,
            Err(e) => {
                return reply_output(format!("Error parsing expression: {}", e));
            }
        };

        // Parse target
        let target_expr =
            match cas_parser::parse(target_str, &mut self.core.engine.simplifier.context) {
                Ok(e) => e,
                Err(e) => {
                    return reply_output(format!("Error parsing target: {}", e));
                }
            };

        // Parse replacement
        let replacement_expr =
            match cas_parser::parse(replacement_str, &mut self.core.engine.simplifier.context) {
                Ok(e) => e,
                Err(e) => {
                    return reply_output(format!("Error parsing replacement: {}", e));
                }
            };

        let mut lines = Vec::new();

        // Detect if target is a simple variable or an expression
        let is_simple_var = target_str.chars().all(|c| c.is_alphanumeric() || c == '_');

        let subbed = if is_simple_var {
            // Variable substitution
            if verbosity != Verbosity::None {
                lines.push(format!(
                    "Variable substitution: {} → {} in {}",
                    target_str, replacement_str, expr_str
                ));
            }
            let target_var = self.core.engine.simplifier.context.var(target_str);
            cas_solver::strategies::substitute_expr(
                &mut self.core.engine.simplifier.context,
                expr,
                target_var,
                replacement_expr,
            )
        } else {
            // Expression substitution (power-aware)
            if verbosity != Verbosity::None {
                lines.push(format!(
                    "Expression substitution: {} → {} in {}",
                    target_str, replacement_str, expr_str
                ));
            }
            cas_engine::substitute::substitute_power_aware(
                &mut self.core.engine.simplifier.context,
                expr,
                target_expr,
                replacement_expr,
                cas_engine::substitute::SubstituteOptions::default(),
            )
        };

        let (result, steps) = self.core.engine.simplifier.simplify(subbed);
        if verbosity != Verbosity::None && !steps.is_empty() {
            if verbosity != Verbosity::Succinct {
                lines.push("Steps:".to_string());
            }
            for step in steps.iter() {
                if should_show_step(step, verbosity) {
                    if verbosity == Verbosity::Succinct {
                        lines.push(format!(
                            "-> {}",
                            DisplayExpr {
                                context: &self.core.engine.simplifier.context,
                                id: step.global_after.unwrap_or(step.after)
                            }
                        ));
                    } else {
                        lines.push(format!("  {}  [{}]", step.description, step.rule_name));
                    }
                }
            }
        }
        lines.push(format!(
            "Result: {}",
            clean_display_string(&format!(
                "{}",
                DisplayExpr {
                    context: &self.core.engine.simplifier.context,
                    id: result
                }
            ))
        ));

        reply_output(lines.join("\n"))
    }

    pub(crate) fn handle_timeline(&mut self, line: &str) {
        let reply = self.handle_timeline_core(line);

        // Post-processing: auto-open timeline.html on macOS after WriteFile is printed
        let has_timeline_write = reply.iter().any(|msg| {
            matches!(msg, ReplMsg::WriteFile { path, .. } if path.to_string_lossy().ends_with("timeline.html"))
        });

        self.print_reply(reply);

        // Try to auto-open on macOS (I/O stays in shell)
        if has_timeline_write {
            #[cfg(target_os = "macos")]
            {
                let _ = std::process::Command::new("open")
                    .arg("timeline.html")
                    .spawn();
            }
        }
    }

    fn handle_timeline_core(&mut self, line: &str) -> ReplReply {
        use std::path::PathBuf;

        let rest = line[9..].trim();

        // Check if the user wants to use "solve" within timeline
        // e.g., "timeline solve x + 2 = 5, x"
        if let Some(solve_rest) = rest.strip_prefix("solve ") {
            // Delegate to solve timeline (still uses old pattern for now)
            self.handle_timeline_solve(solve_rest);
            return ReplReply::new();
        }

        // Check if the user wants to use "simplify" within timeline
        // e.g., "timeline simplify(expr)" or "timeline simplify expr"
        let (expr_str, use_aggressive) = if let Some(inner) = rest
            .strip_prefix("simplify(")
            .and_then(|s| s.strip_suffix(')'))
        {
            // Extract expression from "simplify(expr)"
            (inner, true)
        } else if let Some(simplify_rest) = rest.strip_prefix("simplify ") {
            // Extract expression from "simplify expr"
            (simplify_rest, true)
        } else {
            // No simplify prefix, treat entire rest as expression
            (rest, false)
        };

        // Choose simplifier based on whether aggressive mode is requested
        let (steps, expr_id, simplified) = if use_aggressive {
            // Create temporary simplifier with aggressive rules (like handle_full_simplify)
            let mut temp_simplifier = Simplifier::with_default_rules();
            temp_simplifier.set_collect_steps(true); // Always collect steps for timeline

            // Swap context to preserve variables
            std::mem::swap(
                &mut self.core.engine.simplifier.context,
                &mut temp_simplifier.context,
            );

            match cas_parser::parse(expr_str.trim(), &mut temp_simplifier.context) {
                Ok(expr) => {
                    let (simplified, steps) = temp_simplifier.simplify(expr);

                    // Swap context back
                    std::mem::swap(
                        &mut self.core.engine.simplifier.context,
                        &mut temp_simplifier.context,
                    );

                    // V2.9.9: Apply same pipeline as engine.eval for consistency
                    let display_steps = cas_engine::eval_step_pipeline::to_display_steps(steps);
                    (display_steps, expr, simplified)
                }
                Err(e) => {
                    // Swap context back even on error
                    std::mem::swap(
                        &mut self.core.engine.simplifier.context,
                        &mut temp_simplifier.context,
                    );
                    return reply_output(format!("Parse error: {}", e));
                }
            }
        } else {
            // Use engine.eval like handle_eval does - this ensures the same pipeline
            // (Core → Transform → Rationalize → PostCleanup) is used
            use cas_engine::eval::{EvalAction, EvalRequest, EvalResult};
            use cas_engine::EntryKind;

            // Force collect_steps for timeline
            let was_collecting = self.core.engine.simplifier.collect_steps();
            self.core.engine.simplifier.set_collect_steps(true);

            match cas_parser::parse(expr_str.trim(), &mut self.core.engine.simplifier.context) {
                Ok(expr) => {
                    let req = EvalRequest {
                        raw_input: expr_str.to_string(),
                        parsed: expr,
                        kind: EntryKind::Expr(expr),
                        action: EvalAction::Simplify,
                        auto_store: false, // Don't store in session history for timeline
                    };

                    match self.core.engine.eval(&mut self.core.state, req) {
                        Ok(output) => {
                            let simplified = match output.result {
                                EvalResult::Expr(e) => e,
                                _ => expr, // Fallback
                            };
                            self.core
                                .engine
                                .simplifier
                                .set_collect_steps(was_collecting);
                            (output.steps, expr, simplified)
                        }
                        Err(e) => {
                            self.core
                                .engine
                                .simplifier
                                .set_collect_steps(was_collecting);
                            return reply_output(format!("Simplification error: {}", e));
                        }
                    }
                }
                Err(e) => {
                    self.core
                        .engine
                        .simplifier
                        .set_collect_steps(was_collecting);
                    return reply_output(format!("Parse error: {}", e));
                }
            }
        };

        if steps.is_empty() {
            return reply_output("No simplification steps to visualize.");
        }

        // NOTE: filter_non_productive_steps removed here as timeline already handles filtering
        // and the result was previously unused (prefixed with _)

        // Convert CLI verbosity to timeline verbosity
        // Use Normal level - shows important steps without low-level canonicalization
        let timeline_verbosity = cas_didactic::VerbosityLevel::Normal;

        // Generate HTML timeline with ALL steps and the known simplified result
        // V2.14.40: Pass input string for style preference sniffing (exponential vs radical)
        let mut timeline = cas_didactic::TimelineHtml::new_with_result_and_style(
            &mut self.core.engine.simplifier.context,
            &steps,
            expr_id,
            Some(simplified),
            timeline_verbosity,
            Some(expr_str),
        );
        let html = timeline.to_html();

        // Return WriteFile action + info messages
        let mut reply = ReplReply::new();
        reply.push(ReplMsg::WriteFile {
            path: PathBuf::from("timeline.html"),
            contents: html,
        });
        if use_aggressive {
            reply.push(ReplMsg::output("(Aggressive simplification mode)"));
        }
        reply.push(ReplMsg::output(
            "Open in browser to view interactive visualization.",
        ));
        reply
    }

    pub(crate) fn handle_visualize(&mut self, line: &str) {
        let reply = self.handle_visualize_core(line);
        self.print_reply(reply);
    }

    fn handle_visualize_core(&mut self, line: &str) -> ReplReply {
        use std::path::PathBuf;

        let rest = line
            .strip_prefix("visualize ")
            .or_else(|| line.strip_prefix("viz "))
            .unwrap_or(line)
            .trim();

        match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => {
                let mut viz = cas_engine::visualizer::AstVisualizer::new(
                    &self.core.engine.simplifier.context,
                );
                let dot = viz.to_dot(expr);

                // Return WriteFile action - shell will execute the write
                let filename = PathBuf::from("ast.dot");
                vec![
                    ReplMsg::WriteFile {
                        path: filename,
                        contents: dot,
                    },
                    ReplMsg::output("Render with: dot -Tsvg ast.dot -o ast.svg"),
                    ReplMsg::output("Or: dot -Tpng ast.dot -o ast.png"),
                ]
            }
            Err(e) => reply_output(format!("Parse error: {}", e)),
        }
    }

    pub(crate) fn handle_explain(&mut self, line: &str) {
        let reply = self.handle_explain_core(line);
        self.print_reply(reply);
    }

    fn handle_explain_core(&mut self, line: &str) -> ReplReply {
        let rest = line[8..].trim(); // Remove "explain "

        // Parse the expression
        match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => {
                // Check if it's a function call
                let expr_data = self.core.engine.simplifier.context.get(expr).clone();
                if let Expr::Function(name_id, args) = expr_data {
                    let name = self
                        .core
                        .engine
                        .simplifier
                        .context
                        .sym_name(name_id)
                        .to_string();
                    match name.as_str() {
                        "gcd" => {
                            if args.len() == 2 {
                                // Call the explain_gcd function
                                let result = cas_engine::rules::number_theory::explain_gcd(
                                    &mut self.core.engine.simplifier.context,
                                    args[0],
                                    args[1],
                                );

                                let mut lines = Vec::new();
                                lines.push(format!("Parsed: {}", rest));
                                lines.push(String::new());
                                lines.push("Educational Steps:".to_string());
                                lines.push("─".repeat(60));

                                for step in &result.steps {
                                    lines.push(step.clone());
                                }

                                lines.push("─".repeat(60));
                                lines.push(String::new());

                                if let Some(result_expr) = result.value {
                                    lines.push(format!(
                                        "Result: {}",
                                        clean_display_string(&format!(
                                            "{}",
                                            DisplayExpr {
                                                context: &self.core.engine.simplifier.context,
                                                id: result_expr
                                            }
                                        ))
                                    ));
                                } else {
                                    lines.push("Could not compute GCD".to_string());
                                }
                                reply_output(lines.join("\n"))
                            } else {
                                reply_output("Usage: explain gcd(a, b)")
                            }
                        }
                        _ => reply_output(format!(
                            "Explain mode not yet implemented for function '{}'\n\
                                 Currently supported: gcd",
                            name
                        )),
                    }
                } else {
                    reply_output(
                        "Explain mode currently only supports function calls\n\
                         Try: explain gcd(48, 18)",
                    )
                }
            }
            Err(e) => reply_output(format!("Parse error: {}", e)),
        }
    }
}
