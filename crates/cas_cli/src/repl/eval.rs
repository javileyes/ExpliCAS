use super::*;

impl Repl {
    /// Legacy wrapper - calls core and prints
    pub(crate) fn handle_eval(&mut self, line: &str) {
        let reply = self.handle_eval_core(line);
        self.print_reply(reply);
    }

    /// Core: handle eval command, returns ReplReply (no I/O)
    pub(crate) fn handle_eval_core(&mut self, line: &str) -> ReplReply {
        use cas_ast::root_style::ParseStyleSignals;

        use cas_parser::Statement;
        use cas_solver::{EvalAction, EvalRequest, EvalResult};

        let mut reply: ReplReply = vec![];

        let style_signals = ParseStyleSignals::from_input_string(line);
        let parser_result =
            cas_parser::parse_statement(line, &mut self.core.engine.simplifier.context);

        match parser_result {
            Ok(stmt) => {
                // Map to EvalRequest
                let parsed_expr = match stmt {
                    Statement::Equation(eq) => self
                        .core
                        .engine
                        .simplifier
                        .context
                        .call("Equal", vec![eq.lhs, eq.rhs]),
                    Statement::Expression(e) => e,
                };

                let req = EvalRequest {
                    raw_input: line.to_string(),
                    parsed: parsed_expr,
                    // Eval usually just Simplifies.
                    action: EvalAction::Simplify,
                    auto_store: true,
                };

                match self.core.engine.eval(&mut self.core.state, req) {
                    Ok(output) => {
                        // Display entry number with parsed expression
                        // NOTE: Removed duplicate print! that was causing "#1: #1:" display bug
                        // I'll skip it to reduce noise or rely on Result.
                        // Actually old logic printed: `#{id}  {expr}`.
                        if let Some(id) = output.stored_id {
                            reply.push(ReplMsg::output(format!(
                                "#{}: {}",
                                id,
                                cas_formatter::DisplayExpr {
                                    context: &self.core.engine.simplifier.context,
                                    id: output.parsed
                                }
                            )));
                        }

                        // Show warnings
                        for w in output.domain_warnings {
                            reply.push(ReplMsg::warn(format!(
                                "⚠ {} (from {})",
                                w.message, w.rule_name
                            )));
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

                            let ctx = &self.core.engine.simplifier.context;
                            let display_level = self.core.state.options.requires_display;
                            let debug_mode = self.core.debug_mode;

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
                                reply.push(ReplMsg::info("ℹ️ Requires:".to_string()));
                                // Batch normalize and apply dominance rules
                                let conditions: Vec<_> =
                                    filtered.iter().map(|item| item.cond.clone()).collect();
                                let ctx_mut = &mut self.core.engine.simplifier.context;
                                let normalized_conditions =
                                    cas_solver::normalize_and_dedupe_conditions(
                                        ctx_mut,
                                        &conditions,
                                    );
                                for cond in &normalized_conditions {
                                    if debug_mode {
                                        reply.push(ReplMsg::info(format!(
                                            "  • {} (normalized)",
                                            cond.display(ctx_mut)
                                        )));
                                    } else {
                                        reply.push(ReplMsg::info(format!(
                                            "  • {}",
                                            cond.display(ctx_mut)
                                        )));
                                    }
                                }
                            }
                        }

                        // Collect assumptions from steps for assumption reporting (before steps are consumed)
                        // Deduplicate by (condition_kind, expr_fingerprint) and group by rule
                        let show_assumptions = self.core.state.options.shared.assumption_reporting
                            != cas_solver::AssumptionReporting::Off;
                        let assumed_conditions: Vec<(String, String)> = if show_assumptions {
                            let mut seen: std::collections::HashSet<u64> =
                                std::collections::HashSet::new();
                            let mut result = Vec::new();
                            for step in &output.steps {
                                for event in step.assumption_events() {
                                    // Dedupe by fingerprint
                                    let fp = match &event.key {
                                        cas_solver::AssumptionKey::NonZero { expr_fingerprint } => {
                                            *expr_fingerprint
                                        }
                                        cas_solver::AssumptionKey::Positive {
                                            expr_fingerprint,
                                        } => *expr_fingerprint + 1_000_000,
                                        cas_solver::AssumptionKey::NonNegative {
                                            expr_fingerprint,
                                        } => *expr_fingerprint + 2_000_000,
                                        cas_solver::AssumptionKey::Defined { expr_fingerprint } => {
                                            *expr_fingerprint + 3_000_000
                                        }
                                        cas_solver::AssumptionKey::InvTrigPrincipalRange {
                                            arg_fingerprint,
                                            ..
                                        } => *arg_fingerprint + 4_000_000,
                                        cas_solver::AssumptionKey::ComplexPrincipalBranch {
                                            arg_fingerprint,
                                            ..
                                        } => *arg_fingerprint + 5_000_000,
                                    };
                                    if seen.insert(fp) {
                                        // Format: "x ≠ 0" instead of "≠ 0 (NonZero)"
                                        let condition = match &event.key {
                                            cas_solver::AssumptionKey::NonZero { .. } => {
                                                format!("{} ≠ 0", event.expr_display)
                                            }
                                            cas_solver::AssumptionKey::Positive { .. } => {
                                                format!("{} > 0", event.expr_display)
                                            }
                                            cas_solver::AssumptionKey::NonNegative { .. } => {
                                                format!("{} ≥ 0", event.expr_display)
                                            }
                                            cas_solver::AssumptionKey::Defined { .. } => {
                                                format!("{} is defined", event.expr_display)
                                            }
                                            cas_solver::AssumptionKey::InvTrigPrincipalRange {
                                                func,
                                                ..
                                            } => {
                                                format!(
                                                    "{} in {} principal range",
                                                    event.expr_display, func
                                                )
                                            }
                                            cas_solver::AssumptionKey::ComplexPrincipalBranch {
                                                func,
                                                ..
                                            } => {
                                                format!(
                                                    "{}({}) principal branch",
                                                    func, event.expr_display
                                                )
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
                        // NOTE: show_simplification_steps still prints directly - this is intentional
                        // for now as it's a complex function with its own step formatting logic
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
                            &self.core.engine.simplifier.context,
                            output.parsed,
                            Some(&style_signals),
                        );

                        match output.result {
                            EvalResult::Expr(res) => {
                                // Check if it is Equal function
                                let context = &self.core.engine.simplifier.context;
                                if let Expr::Function(name, args) = context.get(res) {
                                    if context.is_builtin(*name, cas_ast::BuiltinFn::Equal)
                                        && args.len() == 2
                                    {
                                        reply.push(ReplMsg::output(format!(
                                            "Result: {} = {}",
                                            clean_display_string(&format!(
                                                "{}",
                                                cas_formatter::DisplayExprStyled::new(
                                                    context,
                                                    args[0],
                                                    &style_prefs
                                                )
                                            )),
                                            clean_display_string(&format!(
                                                "{}",
                                                cas_formatter::DisplayExprStyled::new(
                                                    context,
                                                    args[1],
                                                    &style_prefs
                                                )
                                            ))
                                        )));
                                        return reply;
                                    }
                                }

                                reply.push(ReplMsg::output(format!(
                                    "Result: {}",
                                    display_expr_or_poly(context, res)
                                )));
                            }
                            EvalResult::SolutionSet(ref solution_set) => {
                                // V2.0: Display full solution set
                                let ctx = &self.core.engine.simplifier.context;
                                reply.push(ReplMsg::output(format!(
                                    "Result: {}",
                                    display_solution_set(ctx, solution_set)
                                )));
                            }
                            EvalResult::Set(_sols) => {
                                reply.push(ReplMsg::output("Result: Set(...)".to_string()));
                                // Simplify result logic doesn't usually produce Set
                            }
                            EvalResult::Bool(b) => {
                                reply.push(ReplMsg::output(format!("Result: {}", b)));
                            }
                            EvalResult::None => {}
                        }

                        // Display blocked hints (pedagogical warnings for Generic mode)
                        // Respects hints_enabled option (can be toggled with `semantics set hints off`)
                        // NOTE: Use output.blocked_hints which were collected by eval() from thread-local
                        // Filter out spurious 'defined' hints when result is undefined
                        // (these are cycle detection artifacts, not useful when result is already undefined)
                        let result_is_undefined = matches!(
                            self.core.engine.simplifier.context.get(output.resolved),
                            cas_ast::Expr::Constant(cas_ast::Constant::Undefined)
                        );
                        let hints: Vec<_> = output
                            .blocked_hints
                            .iter()
                            .filter(|h| !(result_is_undefined && h.key.kind() == "defined"))
                            .collect();
                        if !hints.is_empty() && self.core.state.options.hints_enabled {
                            let ctx = &self.core.engine.simplifier.context;

                            // Helper to format condition with expression
                            let format_condition = |hint: &cas_solver::BlockedHint| -> String {
                                let expr_str = cas_formatter::DisplayExpr {
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
                            let suggestion = match self.core.state.options.shared.semantics.domain_mode {
                                cas_solver::DomainMode::Strict => {
                                    "use `domain generic` or `domain assume` to allow"
                                }
                                cas_solver::DomainMode::Generic => {
                                    "use `semantics set domain assume` to allow analytic assumptions"
                                }
                                cas_solver::DomainMode::Assume => {
                                    // Should not happen, but fallback
                                    "assumptions already enabled"
                                }
                            };

                            // Display grouped hints
                            if grouped.len() == 1 && hints.len() == 1 {
                                // Single hint: compact format
                                let hint = &hints[0];
                                reply.push(ReplMsg::info(format!(
                                    "ℹ️  Blocked: requires {} [{}]",
                                    format_condition(hint),
                                    hint.rule
                                )));
                                reply.push(ReplMsg::info(format!("   {}", suggestion)));
                            } else {
                                // Multiple hints or multiple rules: grouped format
                                reply.push(ReplMsg::info(
                                    "ℹ️  Some simplifications were blocked:".to_string(),
                                ));
                                for (rule, conditions) in &grouped {
                                    if conditions.len() == 1 {
                                        reply.push(ReplMsg::info(format!(
                                            " - Requires {}  [{}]",
                                            conditions[0], rule
                                        )));
                                    } else {
                                        // Compact multiple conditions for same rule
                                        reply.push(ReplMsg::info(format!(
                                            " - Requires {}  [{}]",
                                            conditions.join(", "),
                                            rule
                                        )));
                                    }
                                }
                                reply.push(ReplMsg::info(format!("   Tip: {}", suggestion)));
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
                                reply.push(ReplMsg::info(format!(
                                    "ℹ️  Assumptions used (assumed): {} [{}]",
                                    cond, rule
                                )));
                            } else {
                                reply.push(ReplMsg::info(
                                    "ℹ️  Assumptions used (assumed):".to_string(),
                                ));
                                for (rule, conds) in &by_rule {
                                    reply.push(ReplMsg::info(format!(
                                        "   - {} [{}]",
                                        conds.join(", "),
                                        rule
                                    )));
                                }
                            }
                        }
                    }
                    Err(e) => reply.push(ReplMsg::error(format!("Error: {}", e))),
                }
            }
            Err(e) => reply.push(ReplMsg::error(super::error_render::render_parse_error(
                line, &e,
            ))),
        }

        reply
    }
}
