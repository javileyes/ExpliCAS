use super::*;

impl Repl {
    pub(crate) fn handle_equiv(&mut self, line: &str) {
        use cas_ast::Expr;
        use cas_parser::Statement;

        let rest = line[6..].trim();
        if let Some((expr1_str, expr2_str)) = rsplit_ignoring_parens(rest, ',') {
            let expr1_str = expr1_str.trim();
            let expr2_str = expr2_str.trim();

            // Helper to parse string to ExprId
            // Note: We use a block to limit mutable borrow scope if needed, though sequential calls work for separate statements.
            // But we need to use 'self' to access context inside logic? No, passed as arg.
            fn parse_arg(s: &str, ctx: &mut cas_ast::Context) -> Result<cas_ast::ExprId, String> {
                if s.starts_with('#') && s[1..].chars().all(char::is_numeric) {
                    Ok(ctx.add(Expr::Variable(s.to_string())))
                } else {
                    match cas_parser::parse_statement(s, ctx) {
                        Ok(Statement::Equation(eq)) => {
                            Ok(ctx.add(Expr::Function("Equal".to_string(), vec![eq.lhs, eq.rhs])))
                        }
                        Ok(Statement::Expression(e)) => Ok(e),
                        Err(e) => Err(format!("{}", e)),
                    }
                }
            }

            let e1_res = parse_arg(expr1_str, &mut self.engine.simplifier.context);
            // Verify e1_res to avoid borrow issues? No, Result doesn't borrow.
            let e2_res = parse_arg(expr2_str, &mut self.engine.simplifier.context);

            match (e1_res, e2_res) {
                (Ok(e1), Ok(e2)) => {
                    // V2.14.45: Use new tri-state equivalence check
                    use cas_engine::EquivalenceResult;

                    let result = self.engine.simplifier.are_equivalent_extended(e1, e2);

                    match result {
                        EquivalenceResult::True => {
                            println!("True");
                        }
                        EquivalenceResult::ConditionalTrue { requires } => {
                            println!("True (conditional)");
                            if !requires.is_empty() {
                                println!("ℹ️ Requires:");
                                for req in &requires {
                                    println!("  • {}", req);
                                }
                            }
                        }
                        EquivalenceResult::False => {
                            println!("False");
                        }
                        EquivalenceResult::Unknown => {
                            println!("Unknown (cannot prove equivalence)");
                        }
                    }
                }
                (Err(e), _) => println!("Error parsing first arg: {}", e),
                (_, Err(e)) => println!("Error parsing second arg: {}", e),
            }
        } else {
            println!("Usage: equiv <expr1>, <expr2>");
        }
    }

    pub(crate) fn handle_subst(&mut self, line: &str) {
        // Format: subst <expr>, <target>, <replacement>
        // Examples:
        //   subst x^4 + x^2 + 1, x^2, y   → y² + y + 1 (power-aware)
        //   subst x^2 + x, x, 3          → 12 (variable substitution)
        let rest = line[6..].trim();

        // Split by commas (respecting parentheses)
        let parts: Vec<&str> = split_by_comma_ignoring_parens(rest);

        if parts.len() != 3 {
            println!("Usage: subst <expr>, <target>, <replacement>");
            println!();
            println!("Examples:");
            println!("  subst x^2 + x, x, 3              → 12");
            println!("  subst x^4 + x^2 + 1, x^2, y      → y² + y + 1");
            println!("  subst x^3, x^2, y                → y·x");
            return;
        }

        let expr_str = parts[0].trim();
        let target_str = parts[1].trim();
        let replacement_str = parts[2].trim();

        // Parse the main expression
        let expr = match cas_parser::parse(expr_str, &mut self.engine.simplifier.context) {
            Ok(e) => e,
            Err(e) => {
                println!("Error parsing expression: {}", e);
                return;
            }
        };

        // Parse target
        let target_expr = match cas_parser::parse(target_str, &mut self.engine.simplifier.context) {
            Ok(e) => e,
            Err(e) => {
                println!("Error parsing target: {}", e);
                return;
            }
        };

        // Parse replacement
        let replacement_expr =
            match cas_parser::parse(replacement_str, &mut self.engine.simplifier.context) {
                Ok(e) => e,
                Err(e) => {
                    println!("Error parsing replacement: {}", e);
                    return;
                }
            };

        // Detect if target is a simple variable or an expression
        let is_simple_var = target_str.chars().all(|c| c.is_alphanumeric() || c == '_');

        let subbed = if is_simple_var {
            // Variable substitution
            if self.verbosity != Verbosity::None {
                println!(
                    "Variable substitution: {} → {} in {}",
                    target_str, replacement_str, expr_str
                );
            }
            let target_var = self.engine.simplifier.context.var(target_str);
            cas_engine::solver::strategies::substitute_expr(
                &mut self.engine.simplifier.context,
                expr,
                target_var,
                replacement_expr,
            )
        } else {
            // Expression substitution (power-aware)
            if self.verbosity != Verbosity::None {
                println!(
                    "Expression substitution: {} → {} in {}",
                    target_str, replacement_str, expr_str
                );
            }
            cas_engine::substitute::substitute_power_aware(
                &mut self.engine.simplifier.context,
                expr,
                target_expr,
                replacement_expr,
                cas_engine::substitute::SubstituteOptions::default(),
            )
        };

        let (result, steps) = self.engine.simplifier.simplify(subbed);
        if self.verbosity != Verbosity::None && !steps.is_empty() {
            if self.verbosity != Verbosity::Succinct {
                println!("Steps:");
            }
            for step in steps.iter() {
                if should_show_step(step, self.verbosity) {
                    if self.verbosity == Verbosity::Succinct {
                        println!(
                            "-> {}",
                            DisplayExpr {
                                context: &self.engine.simplifier.context,
                                id: step.global_after.unwrap_or(step.after)
                            }
                        );
                    } else {
                        println!("  {}  [{}]", step.description, step.rule_name);
                    }
                }
            }
        }
        println!(
            "Result: {}",
            clean_display_string(&format!(
                "{}",
                DisplayExpr {
                    context: &self.engine.simplifier.context,
                    id: result
                }
            ))
        );
    }

    pub(crate) fn handle_timeline(&mut self, line: &str) {
        let rest = line[9..].trim();

        // Check if the user wants to use "solve" within timeline
        // e.g., "timeline solve x + 2 = 5, x"
        if let Some(solve_rest) = rest.strip_prefix("solve ") {
            self.handle_timeline_solve(solve_rest);
            return;
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
                &mut self.engine.simplifier.context,
                &mut temp_simplifier.context,
            );

            match cas_parser::parse(expr_str.trim(), &mut temp_simplifier.context) {
                Ok(expr) => {
                    let (simplified, steps) = temp_simplifier.simplify(expr);

                    // Swap context back
                    std::mem::swap(
                        &mut self.engine.simplifier.context,
                        &mut temp_simplifier.context,
                    );

                    // V2.9.9: Apply same pipeline as engine.eval for consistency
                    let display_steps = cas_engine::eval_step_pipeline::to_display_steps(steps);
                    (display_steps, expr, simplified)
                }
                Err(e) => {
                    // Swap context back even on error
                    std::mem::swap(
                        &mut self.engine.simplifier.context,
                        &mut temp_simplifier.context,
                    );
                    println!("Parse error: {}", e);
                    return;
                }
            }
        } else {
            // Use engine.eval like handle_eval does - this ensures the same pipeline
            // (Core → Transform → Rationalize → PostCleanup) is used
            use cas_engine::eval::{EvalAction, EvalRequest, EvalResult};
            use cas_engine::EntryKind;

            // Force collect_steps for timeline
            let was_collecting = self.engine.simplifier.collect_steps();
            self.engine.simplifier.set_collect_steps(true);

            match cas_parser::parse(expr_str.trim(), &mut self.engine.simplifier.context) {
                Ok(expr) => {
                    let req = EvalRequest {
                        raw_input: expr_str.to_string(),
                        parsed: expr,
                        kind: EntryKind::Expr(expr),
                        action: EvalAction::Simplify,
                        auto_store: false, // Don't store in session history for timeline
                    };

                    match self.engine.eval(&mut self.state, req) {
                        Ok(output) => {
                            let simplified = match output.result {
                                EvalResult::Expr(e) => e,
                                _ => expr, // Fallback
                            };
                            self.engine.simplifier.set_collect_steps(was_collecting);
                            (output.steps, expr, simplified)
                        }
                        Err(e) => {
                            self.engine.simplifier.set_collect_steps(was_collecting);
                            println!("Simplification error: {}", e);
                            return;
                        }
                    }
                }
                Err(e) => {
                    self.engine.simplifier.set_collect_steps(was_collecting);
                    println!("Parse error: {}", e);
                    return;
                }
            }
        };

        if steps.is_empty() {
            println!("No simplification steps to visualize.");
            return;
        }

        // NOTE: filter_non_productive_steps removed here as timeline already handles filtering
        // and the result was previously unused (prefixed with _)

        // Convert CLI verbosity to timeline verbosity
        // Use Normal level - shows important steps without low-level canonicalization
        let timeline_verbosity = cas_engine::timeline::VerbosityLevel::Normal;

        // Generate HTML timeline with ALL steps and the known simplified result
        // V2.14.40: Pass input string for style preference sniffing (exponential vs radical)
        let mut timeline = cas_engine::timeline::TimelineHtml::new_with_result_and_style(
            &mut self.engine.simplifier.context,
            &steps,
            expr_id,
            Some(simplified),
            timeline_verbosity,
            Some(expr_str),
        );
        let html = timeline.to_html();

        let filename = "timeline.html";
        match std::fs::write(filename, &html) {
            Ok(_) => {
                println!("Timeline exported to {}", filename);
                if use_aggressive {
                    println!("(Aggressive simplification mode)");
                }
                println!("Open in browser to view interactive visualization.");

                // Try to auto-open on macOS
                #[cfg(target_os = "macos")]
                {
                    let _ = std::process::Command::new("open").arg(filename).spawn();
                }
            }
            Err(e) => println!("Error writing file: {}", e),
        }
    }

    pub(crate) fn handle_visualize(&mut self, line: &str) {
        let rest = line
            .strip_prefix("visualize ")
            .or_else(|| line.strip_prefix("viz "))
            .unwrap_or(line)
            .trim();

        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                let mut viz =
                    cas_engine::visualizer::AstVisualizer::new(&self.engine.simplifier.context);
                let dot = viz.to_dot(expr);

                // Save to file
                let filename = "ast.dot";
                match std::fs::write(filename, &dot) {
                    Ok(_) => {
                        println!("AST exported to {}", filename);
                        println!("Render with: dot -Tsvg {} -o ast.svg", filename);
                        println!("Or: dot -Tpng {} -o ast.png", filename);
                    }
                    Err(e) => println!("Error writing file: {}", e),
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }

    pub(crate) fn handle_explain(&mut self, line: &str) {
        let rest = line[8..].trim(); // Remove "explain "

        // Parse the expression
        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                // Check if it's a function call
                let expr_data = self.engine.simplifier.context.get(expr).clone();
                if let Expr::Function(name, args) = expr_data {
                    match name.as_str() {
                        "gcd" => {
                            if args.len() == 2 {
                                // Call the explain_gcd function
                                let result = cas_engine::rules::number_theory::explain_gcd(
                                    &mut self.engine.simplifier.context,
                                    args[0],
                                    args[1],
                                );

                                println!("Parsed: {}", rest);
                                println!();
                                println!("Educational Steps:");
                                println!("{}", "─".repeat(60));

                                for step in &result.steps {
                                    println!("{}", step);
                                }

                                println!("{}", "─".repeat(60));
                                println!();

                                if let Some(result_expr) = result.value {
                                    println!(
                                        "Result: {}",
                                        clean_display_string(&format!(
                                            "{}",
                                            DisplayExpr {
                                                context: &self.engine.simplifier.context,
                                                id: result_expr
                                            }
                                        ))
                                    );
                                } else {
                                    println!("Could not compute GCD");
                                }
                            } else {
                                println!("Usage: explain gcd(a, b)");
                            }
                        }
                        _ => {
                            println!("Explain mode not yet implemented for function '{}'", name);
                            println!("Currently supported: gcd");
                        }
                    }
                } else {
                    println!("Explain mode currently only supports function calls");
                    println!("Try: explain gcd(48, 18)");
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }

    // ========== SESSION ENVIRONMENT HANDLERS ==========

    /// Handle "let <name> = <expr>" (eager) or "let <name> := <expr>" (lazy) command
    pub(crate) fn handle_let_command(&mut self, rest: &str) {
        // Detect := (lazy) before = (eager) - order matters!
        if let Some(idx) = rest.find(":=") {
            let name = rest[..idx].trim();
            let expr_str = rest[idx + 2..].trim();
            self.handle_assignment(name, expr_str, true); // lazy
        } else if let Some(eq_idx) = rest.find('=') {
            let name = rest[..eq_idx].trim();
            let expr_str = rest[eq_idx + 1..].trim();
            self.handle_assignment(name, expr_str, false); // eager
        } else {
            println!("Usage: let <name> = <expr>   (eager - evaluates)");
            println!("       let <name> := <expr>  (lazy - stores formula)");
            println!("Example: let a = expand((1+x)^3)");
        }
    }

    /// Handle variable assignment (from "let" or ":=")
    /// - eager=false (=): evaluate then store (unwrap __hold)
    /// - eager=true (:=): store formula without evaluating
    pub(crate) fn handle_assignment(&mut self, name: &str, expr_str: &str, lazy: bool) {
        // Validate name
        if name.is_empty() {
            println!("Error: Variable name cannot be empty");
            return;
        }

        // Check if identifier is valid (alphanumeric + underscore, starts with letter/underscore)
        let starts_with_letter = name
            .chars()
            .next()
            .map(|c| c.is_alphabetic())
            .unwrap_or(false);
        if !starts_with_letter && !name.starts_with('_') {
            println!("Error: Variable name must start with a letter or underscore");
            return;
        }

        // Check reserved names
        if cas_engine::env::is_reserved(name) {
            println!(
                "Error: '{}' is a reserved name and cannot be assigned",
                name
            );
            return;
        }

        // Parse the expression
        match cas_parser::parse(expr_str, &mut self.engine.simplifier.context) {
            Ok(rhs_expr) => {
                // Temporarily remove this binding to prevent self-reference in substitute
                let old_binding = self.state.env.get(name);
                self.state.env.unset(name);

                // Substitute using current environment and session refs
                let rhs_substituted = match self
                    .state
                    .resolve_all(&mut self.engine.simplifier.context, rhs_expr)
                {
                    Ok(r) => r,
                    Err(_) => rhs_expr,
                };

                let result = if lazy {
                    // LAZY (:=): store the expression without evaluating
                    rhs_substituted
                } else {
                    // EAGER (=): simplify the expression, then unwrap __hold
                    let (simplified, _steps) = self.engine.simplifier.simplify(rhs_substituted);

                    // Unwrap top-level __hold to get the actual polynomial
                    unwrap_hold_top(&self.engine.simplifier.context, simplified)
                };

                // Store the binding
                self.state.env.set(name.to_string(), result);

                // Display confirmation (with mode indicator for lazy)
                let display = cas_ast::DisplayExpr {
                    context: &self.engine.simplifier.context,
                    id: result,
                };
                if lazy {
                    println!("{} := {}", name, display);
                } else {
                    println!("{} = {}", name, display);
                }

                // Note: we don't restore old_binding - this is an assignment/update
                let _ = old_binding;
            }
            Err(e) => {
                println!("Parse error: {}", e);
            }
        }
    }

    /// Handle "vars" command - list all variable bindings
    pub(crate) fn handle_vars_command(&self) {
        let bindings = self.state.env.list();
        if bindings.is_empty() {
            println!("No variables defined.");
        } else {
            println!("Variables:");
            for (name, expr_id) in bindings {
                let display = cas_ast::DisplayExpr {
                    context: &self.engine.simplifier.context,
                    id: expr_id,
                };
                println!("  {} = {}", name, display);
            }
        }
    }

    /// Handle "clear" or "clear <names>" command
    pub(crate) fn handle_clear_command(&mut self, line: &str) {
        if line == "clear" {
            // Clear all
            let count = self.state.env.len();
            self.state.env.clear_all();
            if count == 0 {
                println!("No variables to clear.");
            } else {
                println!("Cleared {} variable(s).", count);
            }
        } else {
            // Clear specific variables
            let names: Vec<&str> = line[6..].split_whitespace().collect();
            let mut cleared = 0;
            for name in names {
                if self.state.env.unset(name) {
                    cleared += 1;
                } else {
                    println!("Warning: '{}' was not defined", name);
                }
            }
            if cleared > 0 {
                println!("Cleared {} variable(s).", cleared);
            }
        }
    }

    /// Handle "reset" command - reset entire session
    pub(crate) fn handle_reset_command(&mut self) {
        // Clear session state (history + env)
        self.state.clear();

        // Reset simplifier with new context
        self.engine.simplifier = Simplifier::with_default_rules();

        // Re-register custom rules (same as in new())
        self.engine
            .simplifier
            .add_rule(Box::new(cas_engine::rules::functions::AbsSquaredRule));
        self.engine.simplifier.add_rule(Box::new(EvaluateTrigRule));
        self.engine
            .simplifier
            .add_rule(Box::new(PythagoreanIdentityRule));
        if self.config.trig_angle_sum {
            self.engine.simplifier.add_rule(Box::new(AngleIdentityRule));
        }
        self.engine.simplifier.add_rule(Box::new(TanToSinCosRule));
        if self.config.trig_double_angle {
            self.engine.simplifier.add_rule(Box::new(DoubleAngleRule));
        }
        if self.config.canonicalize_trig_square {
            self.engine.simplifier.add_rule(Box::new(
                cas_engine::rules::trigonometry::CanonicalizeTrigSquareRule,
            ));
        }
        self.engine.simplifier.add_rule(Box::new(EvaluateLogRule));
        self.engine
            .simplifier
            .add_rule(Box::new(ExponentialLogRule));
        self.engine
            .simplifier
            .add_rule(Box::new(SimplifyFractionRule));
        self.engine.simplifier.add_rule(Box::new(ExpandRule));
        self.engine
            .simplifier
            .add_rule(Box::new(cas_engine::rules::algebra::ConservativeExpandRule));

        // Sync config
        self.sync_config_to_simplifier();

        // Reset options
        self.debug_mode = false;
        self.last_stats = None;
        self.health_enabled = false;
        self.last_health_report = None;

        println!("Session reset. Environment and context cleared.");
    }

    /// Handle "reset full" command - reset session AND clear profile cache
    pub(crate) fn handle_reset_full_command(&mut self) {
        // First do normal reset
        self.handle_reset_command();

        // Also clear profile cache
        self.state.profile_cache.clear();

        println!("Profile cache cleared (will rebuild on next eval).");
    }

    /// Handle "cache" command - show status or clear cache
    pub(crate) fn handle_cache_command(&mut self, line: &str) {
        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1).copied() {
            None | Some("status") => {
                // Show cache status
                let count = self.state.profile_cache.len();
                println!("Profile Cache: {} profiles cached", count);
                if count == 0 {
                    println!("  (empty - profiles will be built on first eval)");
                } else {
                    println!("  (profiles are reused across evaluations)");
                }
            }
            Some("clear") => {
                self.state.profile_cache.clear();
                println!("Profile cache cleared.");
            }
            Some(cmd) => {
                println!("Unknown cache command: {}", cmd);
                println!("Usage: cache [status|clear]");
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
                println!("Unknown semantics subcommand: '{}'", other);
                println!("Usage: semantics [set|preset|help|<axis>]");
                println!("  semantics            Show all settings");
                println!("  semantics <axis>     Show one axis (domain|value|branch|inv_trig|const_fold|assumptions|assume_scope|requires)");
                println!("  semantics help       Show help");
                println!("  semantics set ...    Change settings");
                println!("  semantics preset     List/apply presets");
            }
        }
    }

    pub(crate) fn print_semantics(&self) {
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        let domain = match self.simplify_options.domain {
            DomainMode::Strict => "strict",
            DomainMode::Assume => "assume",
            DomainMode::Generic => "generic",
        };

        let value = match self.simplify_options.value_domain {
            ValueDomain::RealOnly => "real",
            ValueDomain::ComplexEnabled => "complex",
        };

        let branch = match self.simplify_options.branch {
            BranchPolicy::Principal => "principal",
        };

        let inv_trig = match self.simplify_options.inv_trig {
            InverseTrigPolicy::Strict => "strict",
            InverseTrigPolicy::PrincipalValue => "principal",
        };

        println!("Semantics:");
        println!("  domain_mode: {}", domain);
        println!("  value_domain: {}", value);

        // Show branch with inactive note if value=real
        if self.simplify_options.value_domain == ValueDomain::RealOnly {
            println!("  branch: {} (inactive: value_domain=real)", branch);
        } else {
            println!("  branch: {}", branch);
        }

        println!("  inv_trig: {}", inv_trig);

        let const_fold = match self.state.options.const_fold {
            cas_engine::const_fold::ConstFoldMode::Off => "off",
            cas_engine::const_fold::ConstFoldMode::Safe => "safe",
        };
        println!("  const_fold: {}", const_fold);

        let assumptions = match self.state.options.assumption_reporting {
            cas_engine::AssumptionReporting::Off => "off",
            cas_engine::AssumptionReporting::Summary => "summary",
            cas_engine::AssumptionReporting::Trace => "trace",
        };
        println!("  assumptions: {}", assumptions);

        // Show assume_scope with inactive note if domain_mode != Assume
        let assume_scope = match self.simplify_options.assume_scope {
            cas_engine::AssumeScope::Real => "real",
            cas_engine::AssumeScope::Wildcard => "wildcard",
        };
        if self.simplify_options.domain != DomainMode::Assume {
            println!(
                "  assume_scope: {} (inactive: domain_mode != assume)",
                assume_scope
            );
        } else {
            println!("  assume_scope: {}", assume_scope);
        }

        // Show hints_enabled
        let hints = if self.state.options.hints_enabled {
            "on"
        } else {
            "off"
        };
        println!("  hints: {}", hints);

        // Show requires display level
        let requires = match self.state.options.requires_display {
            cas_engine::implicit_domain::RequiresDisplayLevel::Essential => "essential",
            cas_engine::implicit_domain::RequiresDisplayLevel::All => "all",
        };
        println!("  requires: {}", requires);
    }

    /// Print status for a single semantic axis with current value and available options
    pub(crate) fn print_axis_status(&self, axis: &str) {
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        match axis {
            "domain" => {
                let current = match self.simplify_options.domain {
                    DomainMode::Strict => "strict",
                    DomainMode::Assume => "assume",
                    DomainMode::Generic => "generic",
                };
                println!("domain: {}", current);
                println!("  Values: strict | generic | assume");
                println!("  strict:  No domain assumptions (x/x stays x/x)");
                println!("  generic: Classic CAS 'almost everywhere' algebra");
                println!("  assume:  Use assumptions with warnings");
            }
            "value" => {
                let current = match self.simplify_options.value_domain {
                    ValueDomain::RealOnly => "real",
                    ValueDomain::ComplexEnabled => "complex",
                };
                println!("value: {}", current);
                println!("  Values: real | complex");
                println!("  real:    ℝ only (sqrt(-1) undefined)");
                println!("  complex: ℂ enabled (sqrt(-1) = i)");
            }
            "branch" => {
                let current = match self.simplify_options.branch {
                    BranchPolicy::Principal => "principal",
                };
                let inactive = self.simplify_options.value_domain == ValueDomain::RealOnly;
                if inactive {
                    println!("branch: {} (inactive: value=real)", current);
                } else {
                    println!("branch: {}", current);
                }
                println!("  Values: principal");
                println!("  principal: Use principal branch for multi-valued functions");
                if inactive {
                    println!("  Note: Only active when value=complex");
                }
            }
            "inv_trig" => {
                let current = match self.simplify_options.inv_trig {
                    InverseTrigPolicy::Strict => "strict",
                    InverseTrigPolicy::PrincipalValue => "principal",
                };
                println!("inv_trig: {}", current);
                println!("  Values: strict | principal");
                println!("  strict:    arctan(tan(x)) unchanged");
                println!("  principal: arctan(tan(x)) → x with warning");
            }
            "const_fold" => {
                let current = match self.state.options.const_fold {
                    cas_engine::const_fold::ConstFoldMode::Off => "off",
                    cas_engine::const_fold::ConstFoldMode::Safe => "safe",
                };
                println!("const_fold: {}", current);
                println!("  Values: off | safe");
                println!("  off:  No constant folding (defer semantic decisions)");
                println!("  safe: Fold literals (2^3 → 8, sqrt(-1) → i if complex)");
            }
            "assumptions" => {
                let current = match self.state.options.assumption_reporting {
                    cas_engine::AssumptionReporting::Off => "off",
                    cas_engine::AssumptionReporting::Summary => "summary",
                    cas_engine::AssumptionReporting::Trace => "trace",
                };
                println!("assumptions: {}", current);
                println!("  Values: off | summary | trace");
                println!("  off:     No assumption reporting");
                println!("  summary: Deduped summary line at end");
                println!("  trace:   Detailed trace (future)");
            }
            "assume_scope" => {
                let current = match self.simplify_options.assume_scope {
                    cas_engine::AssumeScope::Real => "real",
                    cas_engine::AssumeScope::Wildcard => "wildcard",
                };
                let inactive = self.simplify_options.domain != DomainMode::Assume;
                if inactive {
                    println!(
                        "assume_scope: {} (inactive: domain_mode != assume)",
                        current
                    );
                } else {
                    println!("assume_scope: {}", current);
                }
                println!("  Values: real | wildcard");
                println!("  real:     Assume for ℝ, error if ℂ needed");
                println!("  wildcard: Assume for ℝ, residual+warning if ℂ needed");
                if inactive {
                    println!("  Note: Only active when domain_mode=assume");
                }
            }
            "requires" => {
                let current = match self.state.options.requires_display {
                    cas_engine::implicit_domain::RequiresDisplayLevel::Essential => "essential",
                    cas_engine::implicit_domain::RequiresDisplayLevel::All => "all",
                };
                println!("requires: {}", current);
                println!("  Values: essential | all");
                println!("  essential: Only show requires whose witness was consumed");
                println!("  all:       Show all requires including implicit ones");
            }
            _ => {
                println!("Unknown axis: {}", axis);
            }
        }
    }

    pub(crate) fn print_semantics_help(&self) {
        println!("Semantics: Control evaluation semantics");
        println!();
        println!("Usage:");
        println!("  semantics                    Show current settings");
        println!("  semantics set <axis> <val>   Set one axis");
        println!("  semantics set k=v k=v ...    Set multiple axes");
        println!();
        println!("Axes:");
        println!("  domain      strict | generic | assume");
        println!("              strict:  No domain assumptions (x/x stays x/x)");
        println!("              generic: Classic CAS 'almost everywhere' algebra");
        println!("              assume:  Use assumptions with warnings");
        println!();
        println!("  value       real | complex");
        println!("              real:    ℝ only (sqrt(-1) undefined)");
        println!("              complex: ℂ enabled (sqrt(-1) = i)");
        println!();
        println!("  branch      principal");
        println!("              (only active when value=complex)");
        println!();
        println!("  inv_trig    strict | principal");
        println!("              strict:    arctan(tan(x)) unchanged");
        println!("              principal: arctan(tan(x)) → x with warning");
        println!();
        println!("  const_fold  off | safe");
        println!("              off:  No constant folding");
        println!("              safe: Fold literals (2^3 → 8)");
        println!();
        println!("  assume_scope real | wildcard");
        println!("              real:     Assume for ℝ, error if ℂ needed");
        println!("              wildcard: Assume for ℝ, residual+warning if ℂ needed");
        println!("              (only active when domain_mode=assume)");
        println!();
        println!("  requires    essential | all");
        println!("              essential: Show only requires whose witness was consumed");
        println!("              all:       Show all requires including implicit ones");
        println!();
        println!("Examples:");
        println!("  semantics set domain strict");
        println!("  semantics set value complex inv_trig principal");
        println!("  semantics set domain=strict value=complex");
        println!("  semantics set assume_scope wildcard");
        println!();
        println!("Presets:");
        println!("  semantics preset              List available presets");
        println!("  semantics preset <name>       Apply a preset");
        println!("  semantics preset help <name>  Show preset details");
    }

    /// Handle "semantics preset" subcommand
    pub(crate) fn handle_preset(&mut self, args: &[&str]) {
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

        match args.first() {
            None => {
                // List presets
                println!("Available presets:");
                for p in &presets {
                    println!("  {:10} {}", p.name, p.description);
                }
                println!();
                println!("Usage:");
                println!("  semantics preset <name>       Apply preset");
                println!("  semantics preset help <name>  Show preset axes");
            }
            Some(&"help") => {
                // Show preset details
                let name = match args.get(1) {
                    Some(name) => *name,
                    None => {
                        println!("Usage: semantics preset help <name>");
                        println!("Presets: default, strict, complex, school");
                        return;
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
                    println!("{}:", p.name);
                    println!("  domain_mode  = {}", domain_str);
                    println!("  value_domain = {}", value_str);
                    println!("  branch       = principal");
                    println!("  inv_trig     = {}", inv_trig_str);
                    println!("  const_fold   = {}", const_fold_str);
                    println!();
                    println!("Purpose: {}", p.description);
                } else {
                    println!("Unknown preset: '{}'", name);
                    println!("Available: default, strict, complex, school");
                }
            }
            Some(name) => {
                // Apply preset
                if let Some(p) = presets.iter().find(|preset| preset.name == *name) {
                    // Capture old values for diff
                    let old_domain = self.simplify_options.domain;
                    let old_value = self.simplify_options.value_domain;
                    let old_branch = self.simplify_options.branch;
                    let old_inv_trig = self.simplify_options.inv_trig;
                    let old_const_fold = self.state.options.const_fold;

                    // Apply preset
                    self.simplify_options.domain = p.domain;
                    self.simplify_options.value_domain = p.value;
                    self.simplify_options.branch = p.branch;
                    self.simplify_options.inv_trig = p.inv_trig;
                    self.state.options.const_fold = p.const_fold;
                    // Sync to state.options (used by evaluation pipeline)
                    self.state.options.domain_mode = p.domain;
                    self.state.options.value_domain = p.value;
                    self.state.options.branch = p.branch;
                    self.state.options.inv_trig = p.inv_trig;

                    self.sync_config_to_simplifier();

                    println!("Applied preset: {}", p.name);
                    println!("Changes:");

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
                        println!("  domain_mode:  {} → {}", old_str, new_str);
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
                        println!("  value_domain: {} → {}", old_str, new_str);
                        changes += 1;
                    }
                    if old_branch != p.branch {
                        println!("  branch:       principal → principal");
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
                        println!("  inv_trig:     {} → {}", old_str, new_str);
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
                        println!("  const_fold:   {} → {}", old_str, new_str);
                        changes += 1;
                    }
                    if changes == 0 {
                        println!("  (no changes - already at this preset)");
                    }
                } else {
                    println!("Unknown preset: '{}'", name);
                    println!("Available: default, strict, complex, school");
                }
            }
        }
    }
}
