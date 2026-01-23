impl Repl {
    fn handle_eval(&mut self, line: &str) {
        use cas_ast::root_style::ParseStyleSignals;

        use cas_engine::eval::{EvalAction, EvalRequest, EvalResult};
        use cas_engine::EntryKind;
        use cas_parser::Statement;

        let style_signals = ParseStyleSignals::from_input_string(line);
        let parser_result = cas_parser::parse_statement(line, &mut self.engine.simplifier.context);

        match parser_result {
            Ok(stmt) => {
                // Map to EvalRequest
                let (kind, parsed_expr) = match stmt {
                    Statement::Equation(eq) => {
                        let eq_expr = self
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
                    raw_input: line.to_string(),
                    parsed: parsed_expr,
                    kind,
                    // Eval usually just Simplifies.
                    action: EvalAction::Simplify,
                    auto_store: true,
                };

                match self.engine.eval(&mut self.state, req) {
                    Ok(output) => {
                        // Display entry number with parsed expression
                        // NOTE: Removed duplicate print! that was causing "#1: #1:" display bug
                        // I'll skip it to reduce noise or rely on Result.
                        // Actually old logic printed: `#{id}  {expr}`.
                        if let Some(id) = output.stored_id {
                            println!(
                                "#{}: {}",
                                id,
                                cas_ast::DisplayExpr {
                                    context: &self.engine.simplifier.context,
                                    id: output.parsed
                                }
                            );
                        }

                        // Show warnings
                        for w in output.domain_warnings {
                            println!("⚠ {} (from {})", w.message, w.rule_name);
                        }

                        // Show required conditions (implicit domain constraints from input)
                        // These are NOT assumptions - they were already required by the input expression
                        // V2.2+: Use unified diagnostics with origin tracking and witness survival filter
                        if !output.diagnostics.requires.is_empty() {
                            // Get result expression for witness survival check
                            let result_expr = match &output.result {
                                EvalResult::Expr(e) => Some(*e),
                                _ => None,
                            };

                            let ctx = &self.engine.simplifier.context;
                            let display_level = self.state.options.requires_display;
                            let debug_mode = self.debug_mode;

                            // Filter requires based on display level and witness survival
                            let filtered: Vec<_> = if let Some(result) = result_expr {
                                output.diagnostics.filter_requires_for_display(
                                    ctx,
                                    result,
                                    display_level,
                                )
                            } else {
                                // No result expr = show all (can't check witness survival)
                                output.diagnostics.requires.iter().collect()
                            };

                            if !filtered.is_empty() {
                                println!("ℹ️ Requires:");
                                // Batch normalize and apply dominance rules
                                let conditions: Vec<_> =
                                    filtered.iter().map(|item| item.cond.clone()).collect();
                                let ctx_mut = &mut self.engine.simplifier.context;
                                let normalized_conditions =
                                    cas_engine::implicit_domain::normalize_and_dedupe_conditions(
                                        ctx_mut,
                                        &conditions,
                                    );
                                for cond in &normalized_conditions {
                                    if debug_mode {
                                        println!("  • {} (normalized)", cond.display(ctx_mut));
                                    } else {
                                        println!("  • {}", cond.display(ctx_mut));
                                    }
                                }
                            }
                        }

                        // Collect assumptions from steps for assumption reporting (before steps are consumed)
                        // Deduplicate by (condition_kind, expr_fingerprint) and group by rule
                        let show_assumptions = self.state.options.assumption_reporting
                            != cas_engine::AssumptionReporting::Off;
                        let assumed_conditions: Vec<(String, String)> = if show_assumptions {
                            let mut seen: std::collections::HashSet<u64> =
                                std::collections::HashSet::new();
                            let mut result = Vec::new();
                            for step in &output.steps {
                                for event in &step.assumption_events {
                                    // Dedupe by fingerprint
                                    let fp = match &event.key {
                                        cas_engine::assumptions::AssumptionKey::NonZero { expr_fingerprint } => *expr_fingerprint,
                                        cas_engine::assumptions::AssumptionKey::Positive { expr_fingerprint } => *expr_fingerprint + 1_000_000,
                                        cas_engine::assumptions::AssumptionKey::NonNegative { expr_fingerprint } => *expr_fingerprint + 2_000_000,
                                        cas_engine::assumptions::AssumptionKey::Defined { expr_fingerprint } => *expr_fingerprint + 3_000_000,
                                        cas_engine::assumptions::AssumptionKey::InvTrigPrincipalRange { arg_fingerprint, .. } => *arg_fingerprint + 4_000_000,
                                        cas_engine::assumptions::AssumptionKey::ComplexPrincipalBranch { arg_fingerprint, .. } => *arg_fingerprint + 5_000_000,
                                    };
                                    if seen.insert(fp) {
                                        // Format: "x ≠ 0" instead of "≠ 0 (NonZero)"
                                        let condition = match &event.key {
                                            cas_engine::assumptions::AssumptionKey::NonZero { .. } => {
                                                format!("{} ≠ 0", event.expr_display)
                                            }
                                            cas_engine::assumptions::AssumptionKey::Positive { .. } => {
                                                format!("{} > 0", event.expr_display)
                                            }
                                            cas_engine::assumptions::AssumptionKey::NonNegative { .. } => {
                                                format!("{} ≥ 0", event.expr_display)
                                            }
                                            cas_engine::assumptions::AssumptionKey::Defined { .. } => {
                                                format!("{} is defined", event.expr_display)
                                            }
                                            cas_engine::assumptions::AssumptionKey::InvTrigPrincipalRange { func, .. } => {
                                                format!("{} in {} principal range", event.expr_display, func)
                                            }
                                            cas_engine::assumptions::AssumptionKey::ComplexPrincipalBranch { func, .. } => {
                                                format!("{}({}) principal branch", func, event.expr_display)
                                            }
                                        };
                                        let rule = step.rule_name.clone();
                                        result.push((condition, rule));
                                    }
                                }
                            }
                            result
                        } else {
                            Vec::new()
                        };

                        // Show steps using helper
                        // We use output.resolved (input to simplify) and output.steps
                        if !output.steps.is_empty() || self.verbosity != Verbosity::None {
                            // trigger logic if verbosity on
                            self.show_simplification_steps(
                                output.resolved,
                                &output.steps,
                                style_signals.clone(),
                            );
                        }

                        // Show Final Result with style sniffing (root notation preservation)
                        let style_prefs = cas_ast::StylePreferences::from_expression_with_signals(
                            &self.engine.simplifier.context,
                            output.parsed,
                            Some(&style_signals),
                        );

                        match output.result {
                            EvalResult::Expr(res) => {
                                // Check if it is Equal function
                                let context = &self.engine.simplifier.context;
                                if let Expr::Function(name, args) = context.get(res) {
                                    if name == "Equal" && args.len() == 2 {
                                        println!(
                                            "Result: {} = {}",
                                            clean_display_string(&format!(
                                                "{}",
                                                cas_ast::DisplayExprStyled::new(
                                                    context,
                                                    args[0],
                                                    &style_prefs
                                                )
                                            )),
                                            clean_display_string(&format!(
                                                "{}",
                                                cas_ast::DisplayExprStyled::new(
                                                    context,
                                                    args[1],
                                                    &style_prefs
                                                )
                                            ))
                                        );
                                        return;
                                    }
                                }

                                println!("Result: {}", display_expr_or_poly(context, res));
                            }
                            EvalResult::SolutionSet(ref solution_set) => {
                                // V2.0: Display full solution set
                                let ctx = &self.engine.simplifier.context;
                                println!("Result: {}", display_solution_set(ctx, solution_set));
                            }
                            EvalResult::Set(_sols) => {
                                println!("Result: Set(...)"); // Simplify result logic doesn't usually produce Set
                            }
                            EvalResult::Bool(b) => println!("Result: {}", b),
                            EvalResult::None => {}
                        }

                        // Display blocked hints (pedagogical warnings for Generic mode)
                        // Respects hints_enabled option (can be toggled with `semantics set hints off`)
                        // NOTE: Use output.blocked_hints which were collected by eval() from thread-local
                        // Filter out spurious 'defined' hints when result is undefined
                        // (these are cycle detection artifacts, not useful when result is already undefined)
                        let result_is_undefined = matches!(
                            self.engine.simplifier.context.get(output.resolved),
                            cas_ast::Expr::Constant(cas_ast::Constant::Undefined)
                        );
                        let hints: Vec<_> = output
                            .blocked_hints
                            .iter()
                            .filter(|h| !(result_is_undefined && h.key.kind() == "defined"))
                            .collect();
                        if !hints.is_empty() && self.state.options.hints_enabled {
                            let ctx = &self.engine.simplifier.context;

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

                            // Group hints by rule name
                            let mut grouped: std::collections::HashMap<String, Vec<String>> =
                                std::collections::HashMap::new();
                            for hint in &hints {
                                grouped
                                    .entry(hint.rule.clone())
                                    .or_default()
                                    .push(format_condition(hint));
                            }

                            // Contextual suggestion based on current mode
                            let suggestion = match self.state.options.domain_mode {
                                cas_engine::DomainMode::Strict => {
                                    "use `domain generic` or `domain assume` to allow"
                                }
                                cas_engine::DomainMode::Generic => {
                                    "use `semantics set domain assume` to allow analytic assumptions"
                                }
                                cas_engine::DomainMode::Assume => {
                                    // Should not happen, but fallback
                                    "assumptions already enabled"
                                }
                            };

                            // Display grouped hints
                            if grouped.len() == 1 && hints.len() == 1 {
                                // Single hint: compact format
                                let hint = &hints[0];
                                println!(
                                    "ℹ️  Blocked: requires {} [{}]",
                                    format_condition(hint),
                                    hint.rule
                                );
                                println!("   {}", suggestion);
                            } else {
                                // Multiple hints or multiple rules: grouped format
                                println!("ℹ️  Some simplifications were blocked:");
                                for (rule, conditions) in &grouped {
                                    if conditions.len() == 1 {
                                        println!(" - Requires {}  [{}]", conditions[0], rule);
                                    } else {
                                        // Compact multiple conditions for same rule
                                        println!(
                                            " - Requires {}  [{}]",
                                            conditions.join(", "),
                                            rule
                                        );
                                    }
                                }
                                println!("   Tip: {}", suggestion);
                            }
                        }

                        // Show assumptions summary when assumption_reporting is enabled (after hints)
                        if show_assumptions && !assumed_conditions.is_empty() {
                            // Group conditions by rule
                            let mut by_rule: std::collections::HashMap<String, Vec<String>> =
                                std::collections::HashMap::new();
                            for (condition, rule) in &assumed_conditions {
                                by_rule
                                    .entry(rule.clone())
                                    .or_default()
                                    .push(condition.clone());
                            }

                            if assumed_conditions.len() == 1 {
                                let (cond, rule) = &assumed_conditions[0];
                                println!("ℹ️  Assumptions used (assumed): {} [{}]", cond, rule);
                            } else {
                                println!("ℹ️  Assumptions used (assumed):");
                                for (rule, conds) in &by_rule {
                                    println!("   - {} [{}]", conds.join(", "), rule);
                                }
                            }
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }
}
