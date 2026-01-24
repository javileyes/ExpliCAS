use super::*;

impl Repl {
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
        let rest = line[12..].trim(); // Remove "weierstrass "

        if rest.is_empty() {
            println!("Usage: weierstrass <expression>");
            println!("Description: Apply Weierstrass substitution (t = tan(x/2))");
            println!("Transforms:");
            println!("  sin(x) → 2t/(1+t²)");
            println!("  cos(x) → (1-t²)/(1+t²)");
            println!("  tan(x) → 2t/(1-t²)");
            println!("Example: weierstrass sin(x) + cos(x)");
            return;
        }

        // Parse the expression
        match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => {
                use cas_ast::DisplayExpr;
                println!("Parsed: {}", rest);
                println!();

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
                println!("Weierstrass substitution (t = tan(x/2)):");
                println!("  {} → {}", rest, result_str);

                // Try to simplify the result
                println!();
                println!("Simplifying...");
                let (simplified, _steps) = self.core.engine.simplifier.simplify(result);
                let simplified_str = clean_display_string(&format!(
                    "{}",
                    DisplayExpr {
                        context: &self.core.engine.simplifier.context,
                        id: simplified
                    }
                ));
                println!("Result: {}", simplified_str);
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }

    /// Apply Weierstrass substitution recursively to all trig functions
    pub(crate) fn apply_weierstrass_recursive(&mut self, expr: cas_ast::ExprId) -> cas_ast::ExprId {
        use cas_ast::Expr;

        match self.core.engine.simplifier.context.get(expr).clone() {
            Expr::Function(name, args)
                if matches!(name.as_str(), "sin" | "cos" | "tan") && args.len() == 1 =>
            {
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
                    .add(Expr::Function("sin".to_string(), vec![half_arg]));
                let cos_half = self
                    .core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Function("cos".to_string(), vec![half_arg]));
                let t = self
                    .core
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Div(sin_half, cos_half)); // t = tan(x/2)

                // Apply appropriate transformation
                match name.as_str() {
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
                }
            }
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
                    .add(Expr::Function(name.clone(), new_args))
            }
            _ => expr, // Number, Variable, Constant, Matrix - leave as is
        }
    }

    pub(crate) fn handle_timeline_solve(&mut self, rest: &str) {
        // Parse equation and variable: "x + 2 = 5, x" or "x + 2 = 5 x"
        let (eq_str, var) = if let Some((e, v)) = rsplit_ignoring_parens(rest, ',') {
            (e.trim(), v.trim())
        } else {
            // No comma. Try to see if it looks like "eq var"
            if let Some((e, v)) = rsplit_ignoring_parens(rest, ' ') {
                let v_trim = v.trim();
                if !v_trim.is_empty() && v_trim.chars().all(char::is_alphabetic) {
                    (e.trim(), v_trim)
                } else {
                    (rest, "x")
                }
            } else {
                (rest, "x")
            }
        };

        match cas_parser::parse_statement(eq_str, &mut self.core.engine.simplifier.context) {
            Ok(cas_parser::Statement::Equation(eq)) => {
                // Call solver with step collection enabled and semantic options
                self.core.engine.simplifier.set_collect_steps(true);
                let solver_opts = cas_engine::solver::SolverOptions {
                    value_domain: self.core.state.options.value_domain,
                    domain_mode: self.core.state.options.domain_mode,
                    assume_scope: self.core.state.options.assume_scope,
                    budget: self.core.state.options.budget,
                    ..Default::default()
                };

                // V2.9.8: Use type-safe API that includes automatic cleanup
                match cas_engine::solver::solve_with_display_steps(
                    &eq,
                    var,
                    &mut self.core.engine.simplifier,
                    solver_opts,
                ) {
                    Ok((solution_set, display_steps)) => {
                        if display_steps.0.is_empty() {
                            println!("No solving steps to visualize.");
                            println!(
                                "Result: {}",
                                display_solution_set(
                                    &self.core.engine.simplifier.context,
                                    &solution_set
                                )
                            );
                            return;
                        }

                        // Generate HTML timeline for solve steps
                        // Use .0 to access the inner Vec<SolveStep>
                        let mut timeline = cas_engine::timeline::SolveTimelineHtml::new(
                            &mut self.core.engine.simplifier.context,
                            &display_steps.0,
                            &eq,
                            &solution_set,
                            var,
                        );
                        let html = timeline.to_html();

                        let filename = "timeline.html";
                        match std::fs::write(filename, &html) {
                            Ok(_) => {
                                println!("Solve timeline exported to {}", filename);
                                println!(
                                    "Result: {}",
                                    display_solution_set(
                                        &self.core.engine.simplifier.context,
                                        &solution_set
                                    )
                                );
                                println!("Open in browser to view interactive visualization.");

                                // Try to auto-open on macOS
                                #[cfg(target_os = "macos")]
                                {
                                    let _ =
                                        std::process::Command::new("open").arg(filename).spawn();
                                }
                            }
                            Err(e) => println!("Error writing file: {}", e),
                        }
                    }
                    Err(e) => println!("Error solving: {}", e),
                }
            }
            Ok(cas_parser::Statement::Expression(_)) => {
                println!("Error: Expected an equation for solve timeline, got an expression.");
                println!("Usage: timeline solve <equation>, <variable>");
                println!("Example: timeline solve x + 2 = 5, x");
            }
            Err(e) => println!("Error parsing equation: {}", e),
        }
    }

    pub(crate) fn handle_solve(&mut self, line: &str) {
        use cas_ast::{DisplayExpr, Expr};
        use cas_engine::eval::{EvalAction, EvalRequest, EvalResult};
        use cas_engine::EntryKind;
        use cas_parser::Statement;

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
        let (eq_str, var) = if let Some((e, v)) = rsplit_ignoring_parens(rest, ',') {
            (e.trim(), v.trim())
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
                    (e_trim, v_trim)
                } else {
                    (rest, "x")
                }
            } else {
                (rest, "x")
            }
        };

        // Parse equation part
        // Style signals handled during display logic mostly, removing invalid context access

        // Handle #id manually as Variable to let Engine resolve it, or parse string
        let parsed_expr_res =
            if eq_str.starts_with('#') && eq_str[1..].chars().all(char::is_numeric) {
                // Pass as Variable("#id") - the engine will now handle this resolution!
                Ok(Statement::Expression(
                    self.core
                        .engine
                        .simplifier
                        .context
                        .add(Expr::Variable(eq_str.to_string())),
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
                            .add(Expr::Function("Equal".to_string(), vec![eq.lhs, eq.rhs]));
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

                let req = EvalRequest {
                    raw_input: eq_str.to_string(),
                    parsed: parsed_expr,
                    kind,
                    action: EvalAction::Solve {
                        var: var.to_string(),
                    },
                    auto_store: true,
                };

                match self.core.engine.eval(&mut self.core.state, req) {
                    Ok(output) => {
                        // Show ID
                        if let Some(id) = output.stored_id {
                            print!("#{}: ", id);
                        }
                        println!("Solving for {}...", var);

                        for w in &output.domain_warnings {
                            println!("⚠ {} (from {})", w.message, w.rule_name);
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
                            println!("⚠ Assumptions: {}", items.join(", "));
                        }

                        // Show Solve Steps
                        // V2.3.8: Three verbosity levels for solver steps:
                        // - None: skip solver steps entirely (just result)
                        // - Succinct: compact steps (3: log, collect+factor, divide)
                        // - Normal/Verbose: detailed steps (5: log, expand, move, factor, divide)
                        let show_solve_steps =
                            !output.solve_steps.is_empty() && self.verbosity != Verbosity::None;

                        if show_solve_steps {
                            println!("Steps:");

                            // V2.9.8: Steps are now pre-cleaned by eval.rs via solve_with_display_steps
                            // No manual cleanup needed - type-safe pipeline guarantees processing

                            // Prepare scoped renderer if scopes are present
                            let registry = cas_ast::display_transforms::DisplayTransformRegistry::with_defaults();
                            let has_scopes = !output.output_scopes.is_empty();
                            let renderer = if has_scopes {
                                Some(cas_ast::display_transforms::ScopedRenderer::new(
                                    &self.core.engine.simplifier.context,
                                    &output.output_scopes,
                                    &registry,
                                ))
                            } else {
                                None
                            };

                            for (i, step) in output.solve_steps.iter().enumerate() {
                                println!("{}. {}", i + 1, step.description);
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
                                println!(
                                    "   -> {} {} {}",
                                    lhs_str, step.equation_after.op, rhs_str
                                );

                                // Display substeps with indentation (educational details)
                                if !step.substeps.is_empty() && self.verbosity == Verbosity::Verbose
                                {
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
                                        println!(
                                            "      {}.{}. {}",
                                            i + 1,
                                            j + 1,
                                            substep.description
                                        );
                                        println!(
                                            "          -> {} {} {}",
                                            sub_lhs, substep.equation_after.op, sub_rhs
                                        );
                                    }
                                }
                            }
                        }

                        match output.result {
                            EvalResult::SolutionSet(ref solution_set) => {
                                // V2.0: Display full solution set including Conditional
                                let ctx = &self.core.engine.simplifier.context;
                                println!("Result: {}", display_solution_set(ctx, solution_set));
                            }
                            EvalResult::Set(ref sols) => {
                                // Legacy: discrete solutions as Vec<ExprId>
                                let ctx = &self.core.engine.simplifier.context;
                                let sol_strs: Vec<String> = if !output.output_scopes.is_empty() {
                                    let registry = cas_ast::display_transforms::DisplayTransformRegistry::with_defaults();
                                    let renderer = cas_ast::display_transforms::ScopedRenderer::new(
                                        ctx,
                                        &output.output_scopes,
                                        &registry,
                                    );
                                    sols.iter().map(|id| renderer.render(*id)).collect()
                                } else {
                                    // Standard display without transforms
                                    sols.iter()
                                        .map(|id| {
                                            DisplayExpr {
                                                context: ctx,
                                                id: *id,
                                            }
                                            .to_string()
                                        })
                                        .collect()
                                };
                                if sol_strs.is_empty() {
                                    println!("Result: No solution");
                                } else {
                                    println!("Result: {{ {} }}", sol_strs.join(", "));
                                }
                            }
                            _ => println!("Result: {:?}", output.result),
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
                            println!("ℹ️ Requires:");
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
                                    println!("  • {} (normalized)", cond.display(ctx_mut));
                                } else {
                                    println!("  • {}", cond.display(ctx_mut));
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
                                        var,
                                        solution_set,
                                    );

                                    // Display verification status
                                    match verify_result.summary {
                                        VerifySummary::AllVerified => {
                                            println!("✓ All solutions verified");
                                        }
                                        VerifySummary::PartiallyVerified => {
                                            println!("⚠ Some solutions verified");
                                            for (sol_id, status) in &verify_result.solutions {
                                                let sol_str = DisplayExpr {
                                                    context: &self.core.engine.simplifier.context,
                                                    id: *sol_id,
                                                }
                                                .to_string();
                                                match status {
                                                    cas_engine::solver::check::VerifyStatus::Verified => {
                                                        println!("  ✓ {} = {} verified", var, sol_str);
                                                    }
                                                    cas_engine::solver::check::VerifyStatus::Unverifiable { reason, .. } => {
                                                        println!("  ⚠ {} = {}: {}", var, sol_str, reason);
                                                    }
                                                    cas_engine::solver::check::VerifyStatus::NotCheckable { reason } => {
                                                        println!("  ℹ {} = {}: {}", var, sol_str, reason);
                                                    }
                                                }
                                            }
                                        }
                                        VerifySummary::NoneVerified => {
                                            println!("⚠ No solutions could be verified");
                                        }
                                        VerifySummary::NotCheckable => {
                                            if let Some(desc) = verify_result.guard_description {
                                                println!("ℹ {}", desc);
                                            } else {
                                                println!("ℹ Solution type not checkable");
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
                            println!(); // Separator line
                            let ctx = &self.core.engine.simplifier.context;

                            // Block 1: Assumptions used
                            if has_assumptions {
                                println!("ℹ️ Assumptions used:");
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
                                    println!("  - {}", cond);
                                }
                            }

                            // Block 2: Blocked simplifications
                            if has_blocked {
                                println!("ℹ️ Blocked simplifications:");
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
                                    println!("  - requires {}  [{}]", cond, rule);
                                }

                                // Contextual suggestion
                                let suggestion = match self.core.state.options.domain_mode {
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
                                println!("  {}", suggestion);
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

                            let suggestion = match self.core.state.options.domain_mode {
                                cas_engine::DomainMode::Strict => {
                                    "use `domain generic` or `domain assume` to allow"
                                }
                                cas_engine::DomainMode::Generic => {
                                    "use `semantics set domain assume` to allow"
                                }
                                cas_engine::DomainMode::Assume => "assumptions already enabled",
                            };

                            println!("\nℹ️ Blocked simplifications:");
                            for hint in &hints {
                                println!("  - requires {} [{}]", format_condition(hint), hint.rule);
                            }
                            println!("  tip: {}", suggestion);
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }
}
