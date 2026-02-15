use super::*;

impl Repl {
    /// Infer the variable to solve for when user doesn't specify one explicitly.
    ///
    /// Returns:
    /// - `Ok(Some(var))` if exactly one free variable found
    /// - `Ok(None)` if no variables found (error case)
    /// - `Err(vars)` if multiple variables found (ambiguous)
    ///
    /// Filters out:
    /// - Known constants: pi, π, e, i
    /// - Internal symbols: _* and #*
    fn infer_solve_variable(
        ctx: &cas_ast::Context,
        expr: cas_ast::ExprId,
    ) -> Result<Option<String>, Vec<String>> {
        let all_vars = cas_ast::collect_variables(ctx, expr);

        // Filter out known constants and internal symbols
        let free_vars: Vec<String> = all_vars
            .into_iter()
            .filter(|v| {
                // Filter constants
                let is_constant = matches!(v.as_str(), "pi" | "π" | "e" | "i");
                // Filter internal symbols
                let is_internal = v.starts_with('_') || v.starts_with('#');
                !is_constant && !is_internal
            })
            .collect();

        match free_vars.len() {
            0 => Ok(None),
            // free_vars.len() == 1 from match arm; pop() is infallible here
            1 => Ok(free_vars.into_iter().next()),
            _ => {
                // Sort for stable error messages
                let mut sorted = free_vars;
                sorted.sort();
                Err(sorted)
            }
        }
    }

    pub(crate) fn expand_log_recursive(&mut self, expr: cas_ast::ExprId) -> cas_ast::ExprId {
        use cas_ast::Expr;
        use cas_engine::parent_context::ParentContext;
        use cas_engine::rule::Rule;
        use cas_engine::rules::logarithms::LogExpansionRule;
        use cas_engine::DomainMode;

        // Create a parent context with Assume mode to allow expansion of symbolic variables
        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Assume);

        let rule = LogExpansionRule;

        // Try to apply the rule at this node
        if let Some(rewrite) =
            rule.apply(&mut self.core.engine.simplifier.context, expr, &parent_ctx)
        {
            // Recursively expand the result
            return self.expand_log_recursive(rewrite.new_expr);
        }

        // If rule didn't apply, recurse into children
        let expr_data = self.core.engine.simplifier.context.get(expr).clone();
        match expr_data {
            Expr::Add(l, r) => {
                let new_l = self.expand_log_recursive(l);
                let new_r = self.expand_log_recursive(r);
                self.core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Add(new_l, new_r))
            }
            Expr::Sub(l, r) => {
                let new_l = self.expand_log_recursive(l);
                let new_r = self.expand_log_recursive(r);
                self.core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Sub(new_l, new_r))
            }
            Expr::Mul(l, r) => {
                let new_l = self.expand_log_recursive(l);
                let new_r = self.expand_log_recursive(r);
                self.core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Mul(new_l, new_r))
            }
            Expr::Div(l, r) => {
                let new_l = self.expand_log_recursive(l);
                let new_r = self.expand_log_recursive(r);
                self.core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Div(new_l, new_r))
            }
            Expr::Pow(b, e) => {
                let new_b = self.expand_log_recursive(b);
                let new_e = self.expand_log_recursive(e);
                self.core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Pow(new_b, new_e))
            }
            Expr::Neg(inner) => {
                let new_inner = self.expand_log_recursive(inner);
                self.core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Neg(new_inner))
            }
            Expr::Function(name, args) => {
                let new_args: Vec<_> = args.iter().map(|a| self.expand_log_recursive(*a)).collect();
                self.core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Function(name, new_args))
            }
            // Atoms: Number, Variable, Constant, Matrix, SessionRef - return as-is
            _ => expr,
        }
    }

    /// Handle the 'weierstrass' command for applying Weierstrass substitution
    /// Transforms sin(x), cos(x), tan(x) into rational expressions in t = tan(x/2)
    pub(crate) fn handle_weierstrass(&mut self, line: &str) {
        let reply = self.handle_weierstrass_core(line);
        self.print_reply(reply);
    }

    fn handle_weierstrass_core(&mut self, line: &str) -> ReplReply {
        let rest = line[12..].trim(); // Remove "weierstrass "

        if rest.is_empty() {
            return reply_output(
                "Usage: weierstrass <expression>\n\
                 Description: Apply Weierstrass substitution (t = tan(x/2))\n\
                 Transforms:\n\
                   sin(x) → 2t/(1+t²)\n\
                   cos(x) → (1-t²)/(1+t²)\n\
                   tan(x) → 2t/(1-t²)\n\
                 Example: weierstrass sin(x) + cos(x)",
            );
        }

        // Parse the expression
        match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => {
                use cas_ast::DisplayExpr;

                // Apply Weierstrass substitution recursively
                let result = self.apply_weierstrass_recursive(expr);

                // Display result
                let result_str = format!(
                    "{}",
                    DisplayExpr {
                        context: &self.core.engine.simplifier.context,
                        id: result
                    }
                );

                // Try to simplify the result
                let (simplified, _steps) = self.core.engine.simplifier.simplify(result);
                let simplified_str = clean_display_string(&format!(
                    "{}",
                    DisplayExpr {
                        context: &self.core.engine.simplifier.context,
                        id: simplified
                    }
                ));

                reply_output(format!(
                    "Parsed: {}\n\n\
                     Weierstrass substitution (t = tan(x/2)):\n\
                       {} → {}\n\n\
                     Simplifying...\n\
                     Result: {}",
                    rest, rest, result_str, simplified_str
                ))
            }
            Err(e) => reply_output(format!("Parse error: {}", e)),
        }
    }

    /// Apply Weierstrass substitution recursively to all trig functions
    pub(crate) fn apply_weierstrass_recursive(&mut self, expr: cas_ast::ExprId) -> cas_ast::ExprId {
        use cas_ast::Expr;

        let expr_data = self.core.engine.simplifier.context.get(expr).clone();

        // Check if it's a trig function first
        if let Expr::Function(name_id, ref args) = expr_data {
            let name = self
                .core
                .engine
                .simplifier
                .context
                .sym_name(name_id)
                .to_string();
            if matches!(name.as_str(), "sin" | "cos" | "tan") && args.len() == 1 {
                let arg = args[0];

                // Build t = tan(x/2) as sin(x/2)/cos(x/2)
                let two_num = self.core.engine.simplifier.context.num(2);
                let half_arg = self
                    .core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Div(arg, two_num));
                let sin_half = self
                    .core
                    .engine
                    .simplifier
                    .context
                    .call("sin", vec![half_arg]);
                let cos_half = self
                    .core
                    .engine
                    .simplifier
                    .context
                    .call("cos", vec![half_arg]);
                let t = self
                    .core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Div(sin_half, cos_half)); // t = tan(x/2)

                // Apply appropriate transformation
                return match name.as_str() {
                    "sin" => {
                        // sin(x) → 2t/(1+t²)
                        let two = self.core.engine.simplifier.context.num(2);
                        let one = self.core.engine.simplifier.context.num(1);
                        let t_squared = self.core.engine.simplifier.context.add(Expr::Pow(t, two));
                        let numerator = self.core.engine.simplifier.context.add(Expr::Mul(two, t));
                        let denominator = self
                            .core
                            .engine
                            .simplifier
                            .context
                            .add(Expr::Add(one, t_squared));
                        self.core
                            .engine
                            .simplifier
                            .context
                            .add(Expr::Div(numerator, denominator))
                    }
                    "cos" => {
                        // cos(x) → (1-t²)/(1+t²)
                        let one = self.core.engine.simplifier.context.num(1);
                        let two = self.core.engine.simplifier.context.num(2);
                        let t_squared = self.core.engine.simplifier.context.add(Expr::Pow(t, two));
                        let numerator = self
                            .core
                            .engine
                            .simplifier
                            .context
                            .add(Expr::Sub(one, t_squared));
                        let denominator = self
                            .core
                            .engine
                            .simplifier
                            .context
                            .add(Expr::Add(one, t_squared));
                        self.core
                            .engine
                            .simplifier
                            .context
                            .add(Expr::Div(numerator, denominator))
                    }
                    "tan" => {
                        // tan(x) → 2t/(1-t²)
                        let two = self.core.engine.simplifier.context.num(2);
                        let one = self.core.engine.simplifier.context.num(1);
                        let t_squared = self.core.engine.simplifier.context.add(Expr::Pow(t, two));
                        let numerator = self.core.engine.simplifier.context.add(Expr::Mul(two, t));
                        let denominator = self
                            .core
                            .engine
                            .simplifier
                            .context
                            .add(Expr::Sub(one, t_squared));
                        self.core
                            .engine
                            .simplifier
                            .context
                            .add(Expr::Div(numerator, denominator))
                    }
                    _ => expr,
                };
            }
        }

        // Handle other expression types
        match expr_data {
            Expr::Add(l, r) => {
                let new_l = self.apply_weierstrass_recursive(l);
                let new_r = self.apply_weierstrass_recursive(r);
                self.core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Add(new_l, new_r))
            }
            Expr::Sub(l, r) => {
                let new_l = self.apply_weierstrass_recursive(l);
                let new_r = self.apply_weierstrass_recursive(r);
                self.core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Sub(new_l, new_r))
            }
            Expr::Mul(l, r) => {
                let new_l = self.apply_weierstrass_recursive(l);
                let new_r = self.apply_weierstrass_recursive(r);
                self.core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Mul(new_l, new_r))
            }
            Expr::Div(l, r) => {
                let new_l = self.apply_weierstrass_recursive(l);
                let new_r = self.apply_weierstrass_recursive(r);
                self.core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Div(new_l, new_r))
            }
            Expr::Pow(base, exp) => {
                let new_base = self.apply_weierstrass_recursive(base);
                let new_exp = self.apply_weierstrass_recursive(exp);
                self.core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Pow(new_base, new_exp))
            }
            Expr::Neg(e) => {
                let new_e = self.apply_weierstrass_recursive(e);
                self.core.engine.simplifier.context.add(Expr::Neg(new_e))
            }
            Expr::Function(name, args) => {
                // Recurse into function arguments
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&a| self.apply_weierstrass_recursive(a))
                    .collect();
                self.core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Function(name, new_args))
            }
            _ => expr, // Number, Variable, Constant, Matrix - leave as is
        }
    }

    pub(crate) fn handle_timeline_solve(&mut self, rest: &str) {
        let reply = self.handle_timeline_solve_core(rest);

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

    fn handle_timeline_solve_core(&mut self, rest: &str) -> ReplReply {
        use std::path::PathBuf;

        // Parse equation and variable: "x + 2 = 5, x" or "x + 2 = 5 x"
        // var_explicit is Some if user provided variable, None if we need to infer
        let (eq_str, var_explicit) = if let Some((e, v)) = rsplit_ignoring_parens(rest, ',') {
            (e.trim(), Some(v.trim().to_string()))
        } else {
            // No comma. Try to see if it looks like "eq var"
            if let Some((e, v)) = rsplit_ignoring_parens(rest, ' ') {
                let v_trim = v.trim();
                if !v_trim.is_empty() && v_trim.chars().all(char::is_alphabetic) {
                    (e.trim(), Some(v_trim.to_string()))
                } else {
                    (rest, None)
                }
            } else {
                (rest, None)
            }
        };

        match cas_parser::parse_statement(eq_str, &mut self.core.engine.simplifier.context) {
            Ok(cas_parser::Statement::Equation(eq)) => {
                // Infer variable if not explicitly provided
                let var = match var_explicit {
                    Some(v) => v,
                    None => {
                        // For equations, infer from lhs - rhs
                        let eq_expr = self
                            .core
                            .engine
                            .simplifier
                            .context
                            .add(cas_ast::Expr::Sub(eq.lhs, eq.rhs));
                        let ctx = &self.core.engine.simplifier.context;
                        match Self::infer_solve_variable(ctx, eq_expr) {
                            Ok(Some(v)) => v,
                            Ok(None) => {
                                return reply_output(
                                    "Error: timeline solve found no variable.\n\
                                     Use timeline solve <equation>, <variable>",
                                );
                            }
                            Err(vars) => {
                                return reply_output(format!(
                                    "Error: timeline solve found ambiguous variables {{{}}}.\n\
                                     Use timeline solve <equation>, {}",
                                    vars.join(", "),
                                    vars.first().unwrap_or(&"x".to_string())
                                ));
                            }
                        }
                    }
                };

                // Call solver with step collection enabled and semantic options
                self.core.engine.simplifier.set_collect_steps(true);
                let solver_opts = cas_engine::solver::SolverOptions {
                    value_domain: self.core.state.options.shared.semantics.value_domain,
                    domain_mode: self.core.state.options.shared.semantics.domain_mode,
                    assume_scope: self.core.state.options.shared.semantics.assume_scope,
                    budget: self.core.state.options.budget,
                    ..Default::default()
                };

                // V2.9.8: Use type-safe API that includes automatic cleanup
                match cas_engine::solver::solve_with_display_steps(
                    &eq,
                    &var,
                    &mut self.core.engine.simplifier,
                    solver_opts,
                ) {
                    Ok((solution_set, display_steps, _diagnostics)) => {
                        if display_steps.0.is_empty() {
                            let result_str = display_solution_set(
                                &self.core.engine.simplifier.context,
                                &solution_set,
                            );
                            return reply_output(format!(
                                "No solving steps to visualize.\nResult: {}",
                                result_str
                            ));
                        }

                        // Generate HTML timeline for solve steps
                        // Use .0 to access the inner Vec<SolveStep>
                        let mut timeline = cas_engine::timeline::SolveTimelineHtml::new(
                            &mut self.core.engine.simplifier.context,
                            &display_steps.0,
                            &eq,
                            &solution_set,
                            &var,
                        );
                        let html = timeline.to_html();

                        // Return WriteFile action + result
                        let result_str = display_solution_set(
                            &self.core.engine.simplifier.context,
                            &solution_set,
                        );
                        vec![
                            ReplMsg::WriteFile {
                                path: PathBuf::from("timeline.html"),
                                contents: html,
                            },
                            ReplMsg::output(format!("Result: {}", result_str)),
                            ReplMsg::output("Open in browser to view interactive visualization."),
                        ]
                    }
                    Err(e) => reply_output(format!("Error solving: {}", e)),
                }
            }
            Ok(cas_parser::Statement::Expression(_)) => reply_output(
                "Error: Expected an equation for solve timeline, got an expression.\n\
                     Usage: timeline solve <equation>, <variable>\n\
                     Example: timeline solve x + 2 = 5, x",
            ),
            Err(e) => reply_output(format!("Error parsing equation: {}", e)),
        }
    }

    pub(crate) fn handle_solve(&mut self, line: &str) {
        let reply = self.handle_solve_core(line, self.verbosity);
        self.print_reply(reply);
    }

    fn handle_solve_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        use cas_ast::DisplayExpr;
        use cas_engine::eval::{EvalAction, EvalRequest, EvalResult};
        use cas_engine::EntryKind;
        use cas_parser::Statement;

        let mut lines: Vec<String> = Vec::new();

        // solve [--check] <equation>, <var>
        let rest = line[6..].trim();

        // Parse --check flag (one-shot override)
        let (check_enabled, rest) = if let Some(stripped) = rest.strip_prefix("--check") {
            let after_flag = stripped.trim_start();
            (true, after_flag)
        } else {
            // Use session toggle if no explicit flag
            (self.core.state.options.check_solutions, rest)
        };

        // Split by comma or space to get equation and var
        // var_explicit is Some if user explicitly provided a variable, None if we need to infer
        let (eq_str, var_explicit) = if let Some((e, v)) = rsplit_ignoring_parens(rest, ',') {
            (e.trim(), Some(v.trim().to_string()))
        } else {
            // No comma. Try to see if it looks like "eq var"
            if let Some((e, v)) = rsplit_ignoring_parens(rest, ' ') {
                let e_trim = e.trim();
                let v_trim = v.trim();
                // Check if v is a variable name (alphabetic) AND
                // the remaining equation doesn't end with '=' (which would mean v is the RHS) AND
                // there are no operators after '=' (which would mean v is part of an expression)
                let has_operators_after_eq = if let Some(eq_pos) = e_trim.find('=') {
                    let after_eq = &e_trim[eq_pos + 1..];
                    after_eq.contains('+')
                        || after_eq.contains('-')
                        || after_eq.contains('*')
                        || after_eq.contains('/')
                        || after_eq.contains('^')
                } else {
                    false
                };
                if !v_trim.is_empty()
                    && v_trim.chars().all(char::is_alphabetic)
                    && !e_trim.ends_with('=')
                    && !has_operators_after_eq
                {
                    (e_trim, Some(v_trim.to_string()))
                } else {
                    // No explicit variable - will infer after parsing
                    (rest, None)
                }
            } else {
                // No explicit variable - will infer after parsing
                (rest, None)
            }
        };

        // Parse equation part
        // Style signals handled during display logic mostly, removing invalid context access

        // Handle #id manually as Variable to let Engine resolve it, or parse string
        let parsed_expr_res =
            if eq_str.starts_with('#') && eq_str[1..].chars().all(char::is_numeric) {
                // Pass as Variable("#id") - the engine will now handle this resolution!
                Ok(Statement::Expression(
                    self.core.engine.simplifier.context.var(eq_str),
                ))
            } else {
                cas_parser::parse_statement(eq_str, &mut self.core.engine.simplifier.context)
            };

        match parsed_expr_res {
            Ok(stmt) => {
                // Store equation for potential verification
                let original_equation: Option<cas_ast::Equation> = match &stmt {
                    Statement::Equation(eq) => Some(eq.clone()),
                    Statement::Expression(_) => None,
                };

                let (kind, parsed_expr) = match stmt {
                    Statement::Equation(eq) => {
                        let eq_expr = self
                            .core
                            .engine
                            .simplifier
                            .context
                            .call("Equal", vec![eq.lhs, eq.rhs]);
                        (
                            EntryKind::Eq {
                                lhs: eq.lhs,
                                rhs: eq.rhs,
                            },
                            eq_expr,
                        )
                    }
                    Statement::Expression(e) => (EntryKind::Expr(e), e),
                };

                // Determine the variable to solve for
                let var = match var_explicit {
                    Some(v) => v,
                    None => {
                        // Try to infer the variable from the expression
                        let ctx = &self.core.engine.simplifier.context;
                        match Self::infer_solve_variable(ctx, parsed_expr) {
                            Ok(Some(v)) => v,
                            Ok(None) => {
                                return reply_output(
                                    "Error: solve() found no variable to solve for.\n\
                                     Use solve(expr, x) to specify the variable.",
                                );
                            }
                            Err(vars) => {
                                return reply_output(format!(
                                    "Error: solve() found ambiguous variables {{{}}}.\n\
                                     Use solve(expr, {}) or solve(expr, {{{}}}).",
                                    vars.join(", "),
                                    vars.first().unwrap_or(&"x".to_string()),
                                    vars.join(", ")
                                ));
                            }
                        }
                    }
                };

                let req = EvalRequest {
                    raw_input: eq_str.to_string(),
                    parsed: parsed_expr,
                    kind,
                    action: EvalAction::Solve { var: var.clone() },
                    auto_store: true,
                };

                match self.core.engine.eval(&mut self.core.state, req) {
                    Ok(output) => {
                        // Show ID
                        let id_prefix = if let Some(id) = output.stored_id {
                            format!("#{}: ", id)
                        } else {
                            String::new()
                        };
                        lines.push(format!("{}Solving for {}...", id_prefix, var));

                        for w in &output.domain_warnings {
                            lines.push(format!("⚠ {} (from {})", w.message, w.rule_name));
                        }

                        // Show solver assumptions summary if any
                        if !output.solver_assumptions.is_empty() {
                            let items: Vec<String> = output
                                .solver_assumptions
                                .iter()
                                .map(|a| {
                                    if a.count > 1 {
                                        format!("{}({}) (×{})", a.kind, a.expr, a.count)
                                    } else {
                                        format!("{}({})", a.kind, a.expr)
                                    }
                                })
                                .collect();
                            lines.push(format!("⚠ Assumptions: {}", items.join(", ")));
                        }

                        // Show Solve Steps
                        // V2.3.8: Three verbosity levels for solver steps:
                        // - None: skip solver steps entirely (just result)
                        // - Succinct: compact steps (3: log, collect+factor, divide)
                        // - Normal/Verbose: detailed steps (5: log, expand, move, factor, divide)
                        let show_solve_steps =
                            !output.solve_steps.is_empty() && verbosity != Verbosity::None;

                        if show_solve_steps {
                            lines.push("Steps:".to_string());

                            // V2.9.8: Steps are now pre-cleaned by eval.rs via solve_with_display_steps
                            // No manual cleanup needed - type-safe pipeline guarantees processing

                            // Prepare scoped renderer with style preferences
                            let registry = cas_ast::display_transforms::DisplayTransformRegistry::with_defaults();
                            let style = StylePreferences::default();
                            let has_scopes = !output.output_scopes.is_empty();
                            let renderer = if has_scopes {
                                Some(cas_ast::display_transforms::ScopedRenderer::new(
                                    &self.core.engine.simplifier.context,
                                    &output.output_scopes,
                                    &registry,
                                    &style,
                                ))
                            } else {
                                None
                            };

                            for (i, step) in output.solve_steps.iter().enumerate() {
                                lines.push(format!("{}. {}", i + 1, step.description));
                                // Display equation after step with scoped transforms
                                let ctx = &self.core.engine.simplifier.context;
                                let (lhs_str, rhs_str) = if let Some(ref r) = renderer {
                                    (
                                        r.render(step.equation_after.lhs),
                                        r.render(step.equation_after.rhs),
                                    )
                                } else {
                                    (
                                        DisplayExpr {
                                            context: ctx,
                                            id: step.equation_after.lhs,
                                        }
                                        .to_string(),
                                        DisplayExpr {
                                            context: ctx,
                                            id: step.equation_after.rhs,
                                        }
                                        .to_string(),
                                    )
                                };
                                lines.push(format!(
                                    "   -> {} {} {}",
                                    lhs_str, step.equation_after.op, rhs_str
                                ));

                                // Display substeps with indentation (educational details)
                                if !step.substeps.is_empty() && verbosity == Verbosity::Verbose {
                                    for (j, substep) in step.substeps.iter().enumerate() {
                                        let sub_ctx = &self.core.engine.simplifier.context;
                                        let (sub_lhs, sub_rhs) = (
                                            DisplayExpr {
                                                context: sub_ctx,
                                                id: substep.equation_after.lhs,
                                            }
                                            .to_string(),
                                            DisplayExpr {
                                                context: sub_ctx,
                                                id: substep.equation_after.rhs,
                                            }
                                            .to_string(),
                                        );
                                        lines.push(format!(
                                            "      {}.{}. {}",
                                            i + 1,
                                            j + 1,
                                            substep.description
                                        ));
                                        lines.push(format!(
                                            "          -> {} {} {}",
                                            sub_lhs, substep.equation_after.op, sub_rhs
                                        ));
                                    }
                                }
                            }
                        }

                        match output.result {
                            EvalResult::SolutionSet(ref solution_set) => {
                                // V2.0: Display full solution set including Conditional
                                let ctx = &self.core.engine.simplifier.context;
                                lines.push(format!(
                                    "Result: {}",
                                    display_solution_set(ctx, solution_set)
                                ));
                            }
                            EvalResult::Set(ref sols) => {
                                // Legacy: discrete solutions as Vec<ExprId>
                                let ctx = &self.core.engine.simplifier.context;
                                let sol_strs: Vec<String> = {
                                    let registry = cas_ast::display_transforms::DisplayTransformRegistry::with_defaults();
                                    let style = StylePreferences::default();
                                    let renderer = cas_ast::display_transforms::ScopedRenderer::new(
                                        ctx,
                                        &output.output_scopes,
                                        &registry,
                                        &style,
                                    );
                                    sols.iter().map(|id| renderer.render(*id)).collect()
                                };
                                if sol_strs.is_empty() {
                                    lines.push("Result: No solution".to_string());
                                } else {
                                    lines.push(format!("Result: {{ {} }}", sol_strs.join(", ")));
                                }
                            }
                            _ => lines.push(format!("Result: {:?}", output.result)),
                        }

                        // V2.2+: Show Requires (implicit domain conditions from solver)
                        // Using unified diagnostics with origin tracking
                        // First pass: filter with immutable context
                        let result_expr_id = match &output.result {
                            EvalResult::Expr(e) => *e,
                            EvalResult::Set(v) => *v.first().unwrap_or(&output.resolved),
                            _ => output.resolved,
                        };
                        let display_level = self.core.state.options.requires_display;
                        let requires_to_show = {
                            let ctx = &self.core.engine.simplifier.context;
                            output.diagnostics.filter_requires_for_display(
                                ctx,
                                result_expr_id,
                                display_level,
                            )
                        };

                        if !requires_to_show.is_empty() {
                            lines.push("ℹ️ Requires:".to_string());
                            // Batch normalize and apply dominance rules
                            let conditions: Vec<_> = requires_to_show
                                .iter()
                                .map(|item| item.cond.clone())
                                .collect();
                            let ctx_mut = &mut self.core.engine.simplifier.context;
                            let normalized_conditions =
                                cas_engine::implicit_domain::normalize_and_dedupe_conditions(
                                    ctx_mut,
                                    &conditions,
                                );
                            for cond in &normalized_conditions {
                                if self.core.debug_mode {
                                    lines.push(format!(
                                        "  • {} (normalized)",
                                        cond.display(ctx_mut)
                                    ));
                                } else {
                                    lines.push(format!("  • {}", cond.display(ctx_mut)));
                                }
                            }
                        }

                        // V2.2+: Show Assumed (not in domain_warnings from simplification,
                        // but solver_assumptions if they are in Assume mode)
                        // Note: solver_assumptions already shown above as "⚠ Assumptions:"
                        // This is just for consistency labeling

                        // Issue #5: Solution verification (--check flag)
                        if check_enabled {
                            if let EvalResult::SolutionSet(ref solution_set) = output.result {
                                if let Some(ref eq) = original_equation {
                                    use cas_engine::solver::check::{
                                        verify_solution_set, VerifySummary,
                                    };

                                    let verify_result = verify_solution_set(
                                        &mut self.core.engine.simplifier,
                                        eq,
                                        &var,
                                        solution_set,
                                    );

                                    // Display verification status
                                    match verify_result.summary {
                                        VerifySummary::AllVerified => {
                                            lines.push("✓ All solutions verified".to_string());
                                        }
                                        VerifySummary::PartiallyVerified => {
                                            lines.push("⚠ Some solutions verified".to_string());
                                            for (sol_id, status) in &verify_result.solutions {
                                                let sol_str = DisplayExpr {
                                                    context: &self.core.engine.simplifier.context,
                                                    id: *sol_id,
                                                }
                                                .to_string();
                                                match status {
                                                    cas_engine::solver::check::VerifyStatus::Verified => {
                                                        lines.push(format!("  ✓ {} = {} verified", var, sol_str));
                                                    }
                                                    cas_engine::solver::check::VerifyStatus::Unverifiable { reason, .. } => {
                                                        lines.push(format!("  ⚠ {} = {}: {}", var, sol_str, reason));
                                                    }
                                                    cas_engine::solver::check::VerifyStatus::NotCheckable { reason } => {
                                                        lines.push(format!("  ℹ {} = {}: {}", var, sol_str, reason));
                                                    }
                                                }
                                            }
                                        }
                                        VerifySummary::NoneVerified => {
                                            lines.push(
                                                "⚠ No solutions could be verified".to_string(),
                                            );
                                        }
                                        VerifySummary::NotCheckable => {
                                            if let Some(desc) = verify_result.guard_description {
                                                lines.push(format!("ℹ {}", desc));
                                            } else {
                                                lines.push(
                                                    "ℹ Solution type not checkable".to_string(),
                                                );
                                            }
                                        }
                                        VerifySummary::Empty => {
                                            // Empty set - nothing to verify
                                        }
                                    }
                                }
                            }
                        }

                        // V2.1 Issue #3: Explain mode - structured summary for solve
                        // Collect blocked hints for debug output
                        let hints = cas_engine::domain::take_blocked_hints();
                        let has_assumptions = !output.solver_assumptions.is_empty();
                        let has_blocked = !hints.is_empty();

                        if self.core.debug_mode && (has_assumptions || has_blocked) {
                            lines.push(String::new()); // Separator line
                            let ctx = &self.core.engine.simplifier.context;

                            // Block 1: Assumptions used
                            if has_assumptions {
                                lines.push("ℹ️ Assumptions used:".to_string());
                                // Dedup and stable order by (kind, expr)
                                let mut assumption_items: Vec<_> = output
                                    .solver_assumptions
                                    .iter()
                                    .map(|a| {
                                        let cond_str = match a.kind.as_str() {
                                            "Positive" => format!("{} > 0", a.expr),
                                            "NonZero" => format!("{} ≠ 0", a.expr),
                                            "NonNegative" => format!("{} ≥ 0", a.expr),
                                            _ => format!("{} ({})", a.expr, a.kind),
                                        };
                                        cond_str
                                    })
                                    .collect();
                                // Stable sort and dedup
                                assumption_items.sort();
                                assumption_items.dedup();
                                for cond in assumption_items {
                                    lines.push(format!("  - {}", cond));
                                }
                            }

                            // Block 2: Blocked simplifications
                            if has_blocked {
                                lines.push("ℹ️ Blocked simplifications:".to_string());
                                // Helper to format condition with expression
                                let format_condition = |hint: &cas_engine::BlockedHint| -> String {
                                    let expr_str = cas_ast::DisplayExpr {
                                        context: ctx,
                                        id: hint.expr_id,
                                    }
                                    .to_string();
                                    match hint.key.kind() {
                                        "positive" => format!("{} > 0", expr_str),
                                        "nonzero" => format!("{} ≠ 0", expr_str),
                                        "nonnegative" => format!("{} ≥ 0", expr_str),
                                        _ => format!("{} ({})", expr_str, hint.key.kind()),
                                    }
                                };

                                // Dedup by (condition, rule)
                                let mut blocked_items: Vec<_> = hints
                                    .iter()
                                    .map(|h| (format_condition(h), h.rule.to_string()))
                                    .collect();
                                blocked_items.sort();
                                blocked_items.dedup();
                                for (cond, rule) in blocked_items {
                                    lines.push(format!("  - requires {}  [{}]", cond, rule));
                                }

                                // Contextual suggestion
                                let suggestion =
                                    match self.core.state.options.shared.semantics.domain_mode {
                                        cas_engine::DomainMode::Strict => {
                                            "tip: use `domain generic` or `domain assume` to allow"
                                        }
                                        cas_engine::DomainMode::Generic => {
                                            "tip: use `semantics set domain assume` to allow"
                                        }
                                        cas_engine::DomainMode::Assume => {
                                            "tip: assumptions already enabled"
                                        }
                                    };
                                lines.push(format!("  {}", suggestion));
                            }
                        } else if has_blocked && self.core.state.options.hints_enabled {
                            // Legacy: show blocked hints even without debug_mode if hints_enabled
                            let ctx = &self.core.engine.simplifier.context;
                            let format_condition = |hint: &cas_engine::BlockedHint| -> String {
                                let expr_str = cas_ast::DisplayExpr {
                                    context: ctx,
                                    id: hint.expr_id,
                                }
                                .to_string();
                                match hint.key.kind() {
                                    "positive" => format!("{} > 0", expr_str),
                                    "nonzero" => format!("{} ≠ 0", expr_str),
                                    "nonnegative" => format!("{} ≥ 0", expr_str),
                                    _ => format!("{} ({})", expr_str, hint.key.kind()),
                                }
                            };

                            let suggestion =
                                match self.core.state.options.shared.semantics.domain_mode {
                                    cas_engine::DomainMode::Strict => {
                                        "use `domain generic` or `domain assume` to allow"
                                    }
                                    cas_engine::DomainMode::Generic => {
                                        "use `semantics set domain assume` to allow"
                                    }
                                    cas_engine::DomainMode::Assume => "assumptions already enabled",
                                };

                            lines.push(String::new());
                            lines.push("ℹ️ Blocked simplifications:".to_string());
                            for hint in &hints {
                                lines.push(format!(
                                    "  - requires {} [{}]",
                                    format_condition(hint),
                                    hint.rule
                                ));
                            }
                            lines.push(format!("  tip: {}", suggestion));
                        }
                    }
                    Err(e) => lines.push(format!("Error: {}", e)),
                }
            }
            Err(e) => lines.push(format!("Parse error: {}", e)),
        }

        reply_output(lines.join("\n"))
    }
}
