use super::*;

impl Repl {
    pub(crate) fn handle_equiv(&mut self, line: &str) {
        let reply = self.handle_equiv_core(line);
        self.print_reply(reply);
    }

    fn handle_equiv_core(&mut self, line: &str) -> ReplReply {
        use cas_ast::Expr;
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
                        Ok(Statement::Equation(eq)) => {
                            Ok(ctx.call("Equal", vec![eq.lhs, eq.rhs]))
                        }
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
            cas_engine::solver::strategies::substitute_expr(
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
        let timeline_verbosity = cas_engine::timeline::VerbosityLevel::Normal;

        // Generate HTML timeline with ALL steps and the known simplified result
        // V2.14.40: Pass input string for style preference sniffing (exponential vs radical)
        let mut timeline = cas_engine::timeline::TimelineHtml::new_with_result_and_style(
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
                if let Expr::Function(name, args) = expr_data {
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

    // ========== SESSION ENVIRONMENT HANDLERS ==========

    /// Handle "let <name> = <expr>" (eager) or "let <name> := <expr>" (lazy) command
    pub(crate) fn handle_let_command(&mut self, rest: &str) {
        let reply = self.handle_let_command_core(rest);
        self.print_reply(reply);
    }

    fn handle_let_command_core(&mut self, rest: &str) -> ReplReply {
        // Detect := (lazy) before = (eager) - order matters!
        if let Some(idx) = rest.find(":=") {
            let name = rest[..idx].trim();
            let expr_str = rest[idx + 2..].trim();
            self.handle_assignment_core(name, expr_str, true) // lazy
        } else if let Some(eq_idx) = rest.find('=') {
            let name = rest[..eq_idx].trim();
            let expr_str = rest[eq_idx + 1..].trim();
            self.handle_assignment_core(name, expr_str, false) // eager
        } else {
            reply_output(
                "Usage: let <name> = <expr>   (eager - evaluates)\n\
                        let <name> := <expr>  (lazy - stores formula)\n\
                 Example: let a = expand((1+x)^3)",
            )
        }
    }

    /// Handle variable assignment (from "let" or ":=")
    /// - eager=false (=): evaluate then store (unwrap __hold)
    /// - eager=true (:=): store formula without evaluating
    pub(crate) fn handle_assignment(&mut self, name: &str, expr_str: &str, lazy: bool) {
        let reply = self.handle_assignment_core(name, expr_str, lazy);
        self.print_reply(reply);
    }

    fn handle_assignment_core(&mut self, name: &str, expr_str: &str, lazy: bool) -> ReplReply {
        // Validate name
        if name.is_empty() {
            return reply_output("Error: Variable name cannot be empty");
        }

        // Check if identifier is valid (alphanumeric + underscore, starts with letter/underscore)
        let starts_with_letter = name
            .chars()
            .next()
            .map(|c| c.is_alphabetic())
            .unwrap_or(false);
        if !starts_with_letter && !name.starts_with('_') {
            return reply_output("Error: Variable name must start with a letter or underscore");
        }

        // Check reserved names
        if cas_engine::env::is_reserved(name) {
            return reply_output(format!(
                "Error: '{}' is a reserved name and cannot be assigned",
                name
            ));
        }

        // Parse the expression
        match cas_parser::parse(expr_str, &mut self.core.engine.simplifier.context) {
            Ok(rhs_expr) => {
                // Temporarily remove this binding to prevent self-reference in substitute
                let old_binding = self.core.state.env.get(name);
                self.core.state.env.unset(name);

                // Substitute using current environment and session refs
                let rhs_substituted = match self
                    .core
                    .state
                    .resolve_all(&mut self.core.engine.simplifier.context, rhs_expr)
                {
                    Ok(r) => r,
                    Err(_) => rhs_expr,
                };

                let result = if lazy {
                    // LAZY (:=): store the expression without evaluating
                    rhs_substituted
                } else {
                    // EAGER (=): simplify the expression, then unwrap __hold
                    let (simplified, _steps) =
                        self.core.engine.simplifier.simplify(rhs_substituted);

                    // Unwrap top-level __hold to get the actual polynomial
                    unwrap_hold_top(&self.core.engine.simplifier.context, simplified)
                };

                // Store the binding
                self.core.state.env.set(name.to_string(), result);

                // Display confirmation (with mode indicator for lazy)
                let display = cas_ast::DisplayExpr {
                    context: &self.core.engine.simplifier.context,
                    id: result,
                };

                // Note: we don't restore old_binding - this is an assignment/update
                let _ = old_binding;

                if lazy {
                    reply_output(format!("{} := {}", name, display))
                } else {
                    reply_output(format!("{} = {}", name, display))
                }
            }
            Err(e) => reply_output(format!("Parse error: {}", e)),
        }
    }

    /// Handle "vars" command - list all variable bindings
    pub(crate) fn handle_vars_command(&self) {
        let reply = self.handle_vars_command_core();
        self.print_reply(reply);
    }

    /// Core logic for "vars" command
    fn handle_vars_command_core(&self) -> ReplReply {
        let bindings = self.core.state.env.list();
        if bindings.is_empty() {
            reply_output("No variables defined.")
        } else {
            let mut lines = vec!["Variables:".to_string()];
            for (name, expr_id) in bindings {
                let display = cas_ast::DisplayExpr {
                    context: &self.core.engine.simplifier.context,
                    id: expr_id,
                };
                lines.push(format!("  {} = {}", name, display));
            }
            reply_output(lines.join("\n"))
        }
    }

    /// Handle "clear" or "clear <names>" command
    pub(crate) fn handle_clear_command(&mut self, line: &str) {
        let reply = self.handle_clear_command_core(line);
        self.print_reply(reply);
    }

    /// Core logic for "clear" command
    fn handle_clear_command_core(&mut self, line: &str) -> ReplReply {
        let mut reply = ReplReply::new();
        if line == "clear" {
            // Clear all
            let count = self.core.state.env.len();
            self.core.state.env.clear_all();
            if count == 0 {
                reply.push(ReplMsg::output("No variables to clear."));
            } else {
                reply.push(ReplMsg::output(format!("Cleared {} variable(s).", count)));
            }
        } else {
            // Clear specific variables
            let names: Vec<&str> = line[6..].split_whitespace().collect();
            let mut cleared = 0;
            for name in names {
                if self.core.state.env.unset(name) {
                    cleared += 1;
                } else {
                    reply.push(ReplMsg::output(format!(
                        "Warning: '{}' was not defined",
                        name
                    )));
                }
            }
            if cleared > 0 {
                reply.push(ReplMsg::output(format!("Cleared {} variable(s).", cleared)));
            }
        }
        reply
    }

    /// Handle "reset" command - reset entire session
    pub(crate) fn handle_reset_command(&mut self) {
        let reply = self.handle_reset_command_impl();
        self.print_reply(reply);
    }

    /// Implementation of reset - keeps access to both core and config
    fn handle_reset_command_impl(&mut self) -> ReplReply {
        // Clear session state (history + env)
        self.core.state.clear();

        // Reset simplifier with new context
        self.core.engine.simplifier = Simplifier::with_default_rules();

        // Re-register custom rules (same as in new())
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(cas_engine::rules::functions::AbsSquaredRule));
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(EvaluateTrigRule));
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(PythagoreanIdentityRule));
        if self.config.trig_angle_sum {
            self.core
                .engine
                .simplifier
                .add_rule(Box::new(AngleIdentityRule));
        }
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(TanToSinCosRule));
        if self.config.trig_double_angle {
            self.core
                .engine
                .simplifier
                .add_rule(Box::new(DoubleAngleRule));
        }
        if self.config.canonicalize_trig_square {
            self.core.engine.simplifier.add_rule(Box::new(
                cas_engine::rules::trigonometry::CanonicalizeTrigSquareRule,
            ));
        }
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(EvaluateLogRule));
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(ExponentialLogRule));
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(SimplifyFractionRule));
        self.core.engine.simplifier.add_rule(Box::new(ExpandRule));
        self.core
            .engine
            .simplifier
            .add_rule(Box::new(cas_engine::rules::algebra::ConservativeExpandRule));

        // Sync config
        self.sync_config_to_simplifier();

        // Reset options
        self.core.debug_mode = false;
        self.core.last_stats = None;
        self.core.health_enabled = false;
        self.core.last_health_report = None;

        reply_output("Session reset. Environment and context cleared.")
    }

    /// Handle "reset full" command - reset session AND clear profile cache
    pub(crate) fn handle_reset_full_command(&mut self) {
        // First do normal reset (which prints its message)
        let reset_reply = self.handle_reset_command_impl();
        self.print_reply(reset_reply);

        // Also clear profile cache
        self.core.state.profile_cache.clear();

        self.print_reply(reply_output(
            "Profile cache cleared (will rebuild on next eval).",
        ));
    }

    /// Handle "cache" command - show status or clear cache
    pub(crate) fn handle_cache_command(&mut self, line: &str) {
        let reply = self.handle_cache_command_core(line);
        self.print_reply(reply);
    }

    /// Core logic for "cache" command
    fn handle_cache_command_core(&mut self, line: &str) -> ReplReply {
        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1).copied() {
            None | Some("status") => {
                // Show cache status
                let count = self.core.state.profile_cache.len();
                let mut lines = vec![format!("Profile Cache: {} profiles cached", count)];
                if count == 0 {
                    lines.push("  (empty - profiles will be built on first eval)".to_string());
                } else {
                    lines.push("  (profiles are reused across evaluations)".to_string());
                }
                reply_output(lines.join("\n"))
            }
            Some("clear") => {
                self.core.state.profile_cache.clear();
                reply_output("Profile cache cleared.")
            }
            Some(cmd) => {
                let mut reply = ReplReply::new();
                reply.push(ReplMsg::output(format!("Unknown cache command: {}", cmd)));
                reply.push(ReplMsg::output("Usage: cache [status|clear]"));
                reply
            }
        }
    }

    /// Handle "semantics" command - unified control for semantic axes
    pub(crate) fn handle_semantics(&mut self, line: &str) {
        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "semantics" - show current settings
                self.print_semantics();
            }
            Some(&"help") => {
                self.print_semantics_help();
            }
            Some(&"set") => {
                // Parse remaining args as axis=value pairs or axis value pairs
                self.parse_semantics_set(&args[2..]);
            }
            Some(&"domain") => {
                self.print_axis_status("domain");
            }
            Some(&"value") => {
                self.print_axis_status("value");
            }
            Some(&"branch") => {
                self.print_axis_status("branch");
            }
            Some(&"inv_trig") => {
                self.print_axis_status("inv_trig");
            }
            Some(&"const_fold") => {
                self.print_axis_status("const_fold");
            }
            Some(&"assumptions") => {
                self.print_axis_status("assumptions");
            }
            Some(&"assume_scope") => {
                self.print_axis_status("assume_scope");
            }
            Some(&"requires") => {
                self.print_axis_status("requires");
            }
            Some(&"preset") => {
                self.handle_preset(&args[2..]);
            }
            Some(other) => {
                let error_text = format!(
                    "Unknown semantics subcommand: '{}'\n\
                     Usage: semantics [set|preset|help|<axis>]\n\
                       semantics            Show all settings\n\
                       semantics <axis>     Show one axis (domain|value|branch|inv_trig|const_fold|assumptions|assume_scope|requires)\n\
                       semantics help       Show help\n\
                       semantics set ...    Change settings\n\
                       semantics preset     List/apply presets",
                    other
                );
                self.print_reply(reply_output(error_text));
            }
        }
    }

    pub(crate) fn print_semantics(&self) {
        let reply = self.print_semantics_core();
        self.print_reply(reply);
    }

    fn print_semantics_core(&self) -> ReplReply {
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        let mut lines = Vec::new();

        let domain = match self.core.simplify_options.domain {
            DomainMode::Strict => "strict",
            DomainMode::Assume => "assume",
            DomainMode::Generic => "generic",
        };

        let value = match self.core.simplify_options.value_domain {
            ValueDomain::RealOnly => "real",
            ValueDomain::ComplexEnabled => "complex",
        };

        let branch = match self.core.simplify_options.branch {
            BranchPolicy::Principal => "principal",
        };

        let inv_trig = match self.core.simplify_options.inv_trig {
            InverseTrigPolicy::Strict => "strict",
            InverseTrigPolicy::PrincipalValue => "principal",
        };

        lines.push("Semantics:".to_string());
        lines.push(format!("  domain_mode: {}", domain));
        lines.push(format!("  value_domain: {}", value));

        // Show branch with inactive note if value=real
        if self.core.simplify_options.value_domain == ValueDomain::RealOnly {
            lines.push(format!(
                "  branch: {} (inactive: value_domain=real)",
                branch
            ));
        } else {
            lines.push(format!("  branch: {}", branch));
        }

        lines.push(format!("  inv_trig: {}", inv_trig));

        let const_fold = match self.core.state.options.const_fold {
            cas_engine::const_fold::ConstFoldMode::Off => "off",
            cas_engine::const_fold::ConstFoldMode::Safe => "safe",
        };
        lines.push(format!("  const_fold: {}", const_fold));

        let assumptions = match self.core.state.options.assumption_reporting {
            cas_engine::AssumptionReporting::Off => "off",
            cas_engine::AssumptionReporting::Summary => "summary",
            cas_engine::AssumptionReporting::Trace => "trace",
        };
        lines.push(format!("  assumptions: {}", assumptions));

        // Show assume_scope with inactive note if domain_mode != Assume
        let assume_scope = match self.core.simplify_options.assume_scope {
            cas_engine::AssumeScope::Real => "real",
            cas_engine::AssumeScope::Wildcard => "wildcard",
        };
        if self.core.simplify_options.domain != DomainMode::Assume {
            lines.push(format!(
                "  assume_scope: {} (inactive: domain_mode != assume)",
                assume_scope
            ));
        } else {
            lines.push(format!("  assume_scope: {}", assume_scope));
        }

        // Show hints_enabled
        let hints = if self.core.state.options.hints_enabled {
            "on"
        } else {
            "off"
        };
        lines.push(format!("  hints: {}", hints));

        // Show requires display level
        let requires = match self.core.state.options.requires_display {
            cas_engine::implicit_domain::RequiresDisplayLevel::Essential => "essential",
            cas_engine::implicit_domain::RequiresDisplayLevel::All => "all",
        };
        lines.push(format!("  requires: {}", requires));

        reply_output(lines.join("\n"))
    }

    /// Print status for a single semantic axis with current value and available options
    pub(crate) fn print_axis_status(&self, axis: &str) {
        let reply = self.print_axis_status_core(axis);
        self.print_reply(reply);
    }

    fn print_axis_status_core(&self, axis: &str) -> ReplReply {
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        let mut lines = Vec::new();

        match axis {
            "domain" => {
                let current = match self.core.simplify_options.domain {
                    DomainMode::Strict => "strict",
                    DomainMode::Assume => "assume",
                    DomainMode::Generic => "generic",
                };
                lines.push(format!("domain: {}", current));
                lines.push("  Values: strict | generic | assume".to_string());
                lines.push("  strict:  No domain assumptions (x/x stays x/x)".to_string());
                lines.push("  generic: Classic CAS 'almost everywhere' algebra".to_string());
                lines.push("  assume:  Use assumptions with warnings".to_string());
            }
            "value" => {
                let current = match self.core.simplify_options.value_domain {
                    ValueDomain::RealOnly => "real",
                    ValueDomain::ComplexEnabled => "complex",
                };
                lines.push(format!("value: {}", current));
                lines.push("  Values: real | complex".to_string());
                lines.push("  real:    ℝ only (sqrt(-1) undefined)".to_string());
                lines.push("  complex: ℂ enabled (sqrt(-1) = i)".to_string());
            }
            "branch" => {
                let current = match self.core.simplify_options.branch {
                    BranchPolicy::Principal => "principal",
                };
                let inactive = self.core.simplify_options.value_domain == ValueDomain::RealOnly;
                if inactive {
                    lines.push(format!("branch: {} (inactive: value=real)", current));
                } else {
                    lines.push(format!("branch: {}", current));
                }
                lines.push("  Values: principal".to_string());
                lines.push(
                    "  principal: Use principal branch for multi-valued functions".to_string(),
                );
                if inactive {
                    lines.push("  Note: Only active when value=complex".to_string());
                }
            }
            "inv_trig" => {
                let current = match self.core.simplify_options.inv_trig {
                    InverseTrigPolicy::Strict => "strict",
                    InverseTrigPolicy::PrincipalValue => "principal",
                };
                lines.push(format!("inv_trig: {}", current));
                lines.push("  Values: strict | principal".to_string());
                lines.push("  strict:    arctan(tan(x)) unchanged".to_string());
                lines.push("  principal: arctan(tan(x)) → x with warning".to_string());
            }
            "const_fold" => {
                let current = match self.core.state.options.const_fold {
                    cas_engine::const_fold::ConstFoldMode::Off => "off",
                    cas_engine::const_fold::ConstFoldMode::Safe => "safe",
                };
                lines.push(format!("const_fold: {}", current));
                lines.push("  Values: off | safe".to_string());
                lines.push("  off:  No constant folding (defer semantic decisions)".to_string());
                lines.push("  safe: Fold literals (2^3 → 8, sqrt(-1) → i if complex)".to_string());
            }
            "assumptions" => {
                let current = match self.core.state.options.assumption_reporting {
                    cas_engine::AssumptionReporting::Off => "off",
                    cas_engine::AssumptionReporting::Summary => "summary",
                    cas_engine::AssumptionReporting::Trace => "trace",
                };
                lines.push(format!("assumptions: {}", current));
                lines.push("  Values: off | summary | trace".to_string());
                lines.push("  off:     No assumption reporting".to_string());
                lines.push("  summary: Deduped summary line at end".to_string());
                lines.push("  trace:   Detailed trace (future)".to_string());
            }
            "assume_scope" => {
                let current = match self.core.simplify_options.assume_scope {
                    cas_engine::AssumeScope::Real => "real",
                    cas_engine::AssumeScope::Wildcard => "wildcard",
                };
                let inactive = self.core.simplify_options.domain != DomainMode::Assume;
                if inactive {
                    lines.push(format!(
                        "assume_scope: {} (inactive: domain_mode != assume)",
                        current
                    ));
                } else {
                    lines.push(format!("assume_scope: {}", current));
                }
                lines.push("  Values: real | wildcard".to_string());
                lines.push("  real:     Assume for ℝ, error if ℂ needed".to_string());
                lines.push("  wildcard: Assume for ℝ, residual+warning if ℂ needed".to_string());
                if inactive {
                    lines.push("  Note: Only active when domain_mode=assume".to_string());
                }
            }
            "requires" => {
                let current = match self.core.state.options.requires_display {
                    cas_engine::implicit_domain::RequiresDisplayLevel::Essential => "essential",
                    cas_engine::implicit_domain::RequiresDisplayLevel::All => "all",
                };
                lines.push(format!("requires: {}", current));
                lines.push("  Values: essential | all".to_string());
                lines
                    .push("  essential: Only show requires whose witness was consumed".to_string());
                lines.push("  all:       Show all requires including implicit ones".to_string());
            }
            _ => {
                lines.push(format!("Unknown axis: {}", axis));
            }
        }

        reply_output(lines.join("\n"))
    }

    pub(crate) fn print_semantics_help(&self) {
        let reply = self.print_semantics_help_core();
        self.print_reply(reply);
    }

    fn print_semantics_help_core(&self) -> ReplReply {
        let text = r#"Semantics: Control evaluation semantics

Usage:
  semantics                    Show current settings
  semantics set <axis> <val>   Set one axis
  semantics set k=v k=v ...    Set multiple axes

Axes:
  domain      strict | generic | assume
              strict:  No domain assumptions (x/x stays x/x)
              generic: Classic CAS 'almost everywhere' algebra
              assume:  Use assumptions with warnings

  value       real | complex
              real:    ℝ only (sqrt(-1) undefined)
              complex: ℂ enabled (sqrt(-1) = i)

  branch      principal
              (only active when value=complex)

  inv_trig    strict | principal
              strict:    arctan(tan(x)) unchanged
              principal: arctan(tan(x)) → x with warning

  const_fold  off | safe
              off:  No constant folding
              safe: Fold literals (2^3 → 8)

  assume_scope real | wildcard
              real:     Assume for ℝ, error if ℂ needed
              wildcard: Assume for ℝ, residual+warning if ℂ needed
              (only active when domain_mode=assume)

  requires    essential | all
              essential: Show only requires whose witness was consumed
              all:       Show all requires including implicit ones

Examples:
  semantics set domain strict
  semantics set value complex inv_trig principal
  semantics set domain=strict value=complex
  semantics set assume_scope wildcard

Presets:
  semantics preset              List available presets
  semantics preset <name>       Apply a preset
  semantics preset help <name>  Show preset details"#;
        reply_output(text)
    }

    /// Handle "semantics preset" subcommand
    pub(crate) fn handle_preset(&mut self, args: &[&str]) {
        let reply = self.handle_preset_core(args);
        self.print_reply(reply);
    }

    fn handle_preset_core(&mut self, args: &[&str]) -> ReplReply {
        use cas_engine::const_fold::ConstFoldMode;
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        // Preset definitions: (name, description, domain, value, branch, inv_trig, const_fold)
        struct Preset {
            name: &'static str,
            description: &'static str,
            domain: DomainMode,
            value: ValueDomain,
            branch: BranchPolicy,
            inv_trig: InverseTrigPolicy,
            const_fold: ConstFoldMode,
        }

        let presets = [
            Preset {
                name: "default",
                description: "Reset to engine defaults",
                domain: DomainMode::Generic,
                value: ValueDomain::RealOnly,
                branch: BranchPolicy::Principal,
                inv_trig: InverseTrigPolicy::Strict,
                const_fold: ConstFoldMode::Off,
            },
            Preset {
                name: "strict",
                description: "Conservative real + strict domain",
                domain: DomainMode::Strict,
                value: ValueDomain::RealOnly,
                branch: BranchPolicy::Principal,
                inv_trig: InverseTrigPolicy::Strict,
                const_fold: ConstFoldMode::Off,
            },
            Preset {
                name: "complex",
                description: "Enable ℂ + safe const_fold (sqrt(-1) → i)",
                domain: DomainMode::Generic,
                value: ValueDomain::ComplexEnabled,
                branch: BranchPolicy::Principal,
                inv_trig: InverseTrigPolicy::Strict,
                const_fold: ConstFoldMode::Safe,
            },
            Preset {
                name: "school",
                description: "Real + principal inverse trig (arctan(tan(x)) → x)",
                domain: DomainMode::Generic,
                value: ValueDomain::RealOnly,
                branch: BranchPolicy::Principal,
                inv_trig: InverseTrigPolicy::PrincipalValue,
                const_fold: ConstFoldMode::Off,
            },
        ];

        let mut lines = Vec::new();

        match args.first() {
            None => {
                // List presets
                lines.push("Available presets:".to_string());
                for p in &presets {
                    lines.push(format!("  {:10} {}", p.name, p.description));
                }
                lines.push(String::new());
                lines.push("Usage:".to_string());
                lines.push("  semantics preset <name>       Apply preset".to_string());
                lines.push("  semantics preset help <name>  Show preset axes".to_string());
            }
            Some(&"help") => {
                // Show preset details
                let name = match args.get(1) {
                    Some(name) => *name,
                    None => {
                        lines.push("Usage: semantics preset help <name>".to_string());
                        lines.push("Presets: default, strict, complex, school".to_string());
                        return reply_output(lines.join("\n"));
                    }
                };
                if let Some(p) = presets.iter().find(|p| p.name == name) {
                    let domain_str = match p.domain {
                        DomainMode::Strict => "strict",
                        DomainMode::Generic => "generic",
                        DomainMode::Assume => "assume",
                    };
                    let value_str = match p.value {
                        ValueDomain::RealOnly => "real",
                        ValueDomain::ComplexEnabled => "complex",
                    };
                    let inv_trig_str = match p.inv_trig {
                        InverseTrigPolicy::Strict => "strict",
                        InverseTrigPolicy::PrincipalValue => "principal",
                    };
                    let const_fold_str = match p.const_fold {
                        ConstFoldMode::Off => "off",
                        ConstFoldMode::Safe => "safe",
                    };
                    lines.push(format!("{}:", p.name));
                    lines.push(format!("  domain_mode  = {}", domain_str));
                    lines.push(format!("  value_domain = {}", value_str));
                    lines.push("  branch       = principal".to_string());
                    lines.push(format!("  inv_trig     = {}", inv_trig_str));
                    lines.push(format!("  const_fold   = {}", const_fold_str));
                    lines.push(String::new());
                    lines.push(format!("Purpose: {}", p.description));
                } else {
                    lines.push(format!("Unknown preset: '{}'", name));
                    lines.push("Available: default, strict, complex, school".to_string());
                }
            }
            Some(name) => {
                // Apply preset
                if let Some(p) = presets.iter().find(|preset| preset.name == *name) {
                    // Capture old values for diff
                    let old_domain = self.core.simplify_options.domain;
                    let old_value = self.core.simplify_options.value_domain;
                    let old_branch = self.core.simplify_options.branch;
                    let old_inv_trig = self.core.simplify_options.inv_trig;
                    let old_const_fold = self.core.state.options.const_fold;

                    // Apply preset
                    self.core.simplify_options.domain = p.domain;
                    self.core.simplify_options.value_domain = p.value;
                    self.core.simplify_options.branch = p.branch;
                    self.core.simplify_options.inv_trig = p.inv_trig;
                    self.core.state.options.const_fold = p.const_fold;
                    // Sync to state.options (used by evaluation pipeline)
                    self.core.state.options.domain_mode = p.domain;
                    self.core.state.options.value_domain = p.value;
                    self.core.state.options.branch = p.branch;
                    self.core.state.options.inv_trig = p.inv_trig;

                    self.sync_config_to_simplifier();

                    lines.push(format!("Applied preset: {}", p.name));
                    lines.push("Changes:".to_string());

                    // Print changes
                    let mut changes = 0;
                    if old_domain != p.domain {
                        let old_str = match old_domain {
                            DomainMode::Strict => "strict",
                            DomainMode::Generic => "generic",
                            DomainMode::Assume => "assume",
                        };
                        let new_str = match p.domain {
                            DomainMode::Strict => "strict",
                            DomainMode::Generic => "generic",
                            DomainMode::Assume => "assume",
                        };
                        lines.push(format!("  domain_mode:  {} → {}", old_str, new_str));
                        changes += 1;
                    }
                    if old_value != p.value {
                        let old_str = match old_value {
                            ValueDomain::RealOnly => "real",
                            ValueDomain::ComplexEnabled => "complex",
                        };
                        let new_str = match p.value {
                            ValueDomain::RealOnly => "real",
                            ValueDomain::ComplexEnabled => "complex",
                        };
                        lines.push(format!("  value_domain: {} → {}", old_str, new_str));
                        changes += 1;
                    }
                    if old_branch != p.branch {
                        lines.push("  branch:       principal → principal".to_string());
                        changes += 1;
                    }
                    if old_inv_trig != p.inv_trig {
                        let old_str = match old_inv_trig {
                            InverseTrigPolicy::Strict => "strict",
                            InverseTrigPolicy::PrincipalValue => "principal",
                        };
                        let new_str = match p.inv_trig {
                            InverseTrigPolicy::Strict => "strict",
                            InverseTrigPolicy::PrincipalValue => "principal",
                        };
                        lines.push(format!("  inv_trig:     {} → {}", old_str, new_str));
                        changes += 1;
                    }
                    if old_const_fold != p.const_fold {
                        let old_str = match old_const_fold {
                            ConstFoldMode::Off => "off",
                            ConstFoldMode::Safe => "safe",
                        };
                        let new_str = match p.const_fold {
                            ConstFoldMode::Off => "off",
                            ConstFoldMode::Safe => "safe",
                        };
                        lines.push(format!("  const_fold:   {} → {}", old_str, new_str));
                        changes += 1;
                    }
                    if changes == 0 {
                        lines.push("  (no changes - already at this preset)".to_string());
                    }
                } else {
                    lines.push(format!("Unknown preset: '{}'", name));
                    lines.push("Available: default, strict, complex, school".to_string());
                }
            }
        }

        reply_output(lines.join("\n"))
    }
}
