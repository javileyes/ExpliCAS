use super::*;

impl Repl {
    pub(crate) fn handle_budget_command(&mut self, line: &str) {
        let reply = self.handle_budget_command_core(line);
        self.print_reply(reply);
    }

    fn handle_budget_command_core(&mut self, line: &str) -> ReplReply {
        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "budget" - show current setting
                let budget = self.core.state.options().budget;
                reply_output(format!(
                    "Solve budget: max_branches={}\n\
                      Controls how many case splits the solver can create.\n\
                      0: No splits (fallback to simple solutions)\n\
                      1: Conservative (default)\n\
                      2+: Allow case splits for symbolic bases (a^x=a, etc)\n\
                      (use 'budget N' to change, e.g. 'budget 2')",
                    budget.max_branches
                ))
            }
            Some(n_str) => {
                if let Ok(n) = n_str.parse::<usize>() {
                    self.core.state.options_mut().budget.max_branches = n;
                    let mode_msg = if n == 0 {
                        "  ⚠️ No case splits allowed (fallback to simple solutions)"
                    } else if n == 1 {
                        "  Conservative mode (default)"
                    } else {
                        "  ✓ Case splits enabled for symbolic bases\n  Try: solve a^x = a"
                    };
                    reply_output(format!("Solve budget: max_branches = {}\n{}", n, mode_msg))
                } else {
                    reply_output(format!(
                        "Invalid budget value: '{}' (expected a number)\n\
                         Usage: budget N\n\
                           budget 0  - No case splits\n\
                           budget 1  - Conservative (default)\n\
                           budget 2  - Allow case splits for a^x=a patterns",
                        n_str
                    ))
                }
            }
        }
    }

    /// Handle "history" or "list" command - show session history
    pub(crate) fn handle_history_command(&self) {
        let reply = self.handle_history_command_core();
        self.print_reply(reply);
    }

    fn handle_history_command_core(&self) -> ReplReply {
        let entries = self.core.state.store().list();
        if entries.is_empty() {
            return reply_output("No entries in session history.");
        }

        let mut lines = vec![format!("Session history ({} entries):", entries.len())];
        for entry in entries {
            let type_indicator = match &entry.kind {
                cas_session::EntryKind::Expr(_) => "Expr",
                cas_session::EntryKind::Eq { .. } => "Eq  ",
            };
            // Show simplified form if possible
            let display = match &entry.kind {
                cas_session::EntryKind::Expr(expr_id) => {
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &self.core.engine.simplifier.context,
                            id: *expr_id
                        }
                    )
                }
                cas_session::EntryKind::Eq { lhs, rhs } => {
                    format!(
                        "{} = {}",
                        cas_formatter::DisplayExpr {
                            context: &self.core.engine.simplifier.context,
                            id: *lhs
                        },
                        cas_formatter::DisplayExpr {
                            context: &self.core.engine.simplifier.context,
                            id: *rhs
                        }
                    )
                }
            };
            lines.push(format!(
                "  #{:<3} [{}] {}",
                entry.id, type_indicator, display
            ));
        }
        reply_output(lines.join("\n"))
    }

    /// Handle "show #id" command - show details of a specific entry
    pub(crate) fn handle_show_command(&mut self, line: &str) {
        let reply = self.handle_show_command_core(line);
        self.print_reply(reply);
    }

    fn handle_show_command_core(&mut self, line: &str) -> ReplReply {
        let input = line.trim().trim_start_matches('#');
        match input.parse::<u64>() {
            Ok(id) => {
                // Clone entry data to avoid borrow conflicts
                let entry_data = self
                    .core
                    .state
                    .store()
                    .get(id)
                    .map(|e| (e.type_str().to_string(), e.raw_text.clone(), e.kind.clone()));

                if let Some((type_str, raw_text, kind)) = entry_data {
                    let mut lines = vec![
                        format!("Entry #{}:", id),
                        format!("  Type:       {}", type_str),
                        format!("  Raw:        {}", raw_text),
                    ];

                    match &kind {
                        cas_session::EntryKind::Expr(expr_id) => {
                            // Show parsed expression
                            lines.push(format!(
                                "  Parsed:     {}",
                                DisplayExpr {
                                    context: &self.core.engine.simplifier.context,
                                    id: *expr_id
                                }
                            ));

                            // Show resolved (after #id and env substitution)
                            let resolved = match cas_session::resolve_all_from_state(
                                &mut self.core.engine.simplifier.context,
                                *expr_id,
                                &self.core.state,
                            ) {
                                Ok(r) => r,
                                Err(_) => *expr_id,
                            };
                            if resolved != *expr_id {
                                lines.push(format!(
                                    "  Resolved:   {}",
                                    DisplayExpr {
                                        context: &self.core.engine.simplifier.context,
                                        id: resolved
                                    }
                                ));
                            }

                            // Perform full eval to get requires/assumed metadata
                            let req = cas_solver::EvalRequest {
                                raw_input: raw_text.clone(),
                                parsed: *expr_id,
                                action: cas_solver::EvalAction::Simplify,
                                auto_store: false,
                            };

                            if let Ok(output) = self.core.engine.eval(&mut self.core.state, req) {
                                // Show simplified result
                                if let cas_solver::EvalResult::Expr(simplified) = &output.result {
                                    if *simplified != *expr_id {
                                        lines.push(format!(
                                            "  Simplified: {}",
                                            DisplayExpr {
                                                context: &self.core.engine.simplifier.context,
                                                id: *simplified
                                            }
                                        ));
                                    }
                                }

                                // Show Requires (implicit domain conditions)
                                if !output.required_conditions.is_empty() {
                                    lines.push("  ℹ️ Requires:".to_string());
                                    for cond in &output.required_conditions {
                                        lines.push(format!(
                                            "    - {}",
                                            cond.display(&self.core.engine.simplifier.context)
                                        ));
                                    }
                                }

                                // Show Assumed (domain warnings)
                                if !output.domain_warnings.is_empty() {
                                    lines.push("  ⚠ Assumed:".to_string());
                                    for w in &output.domain_warnings {
                                        lines.push(format!("    - {}", w.message));
                                    }
                                }

                                // Show Blocked hints (rules that couldn't fire)
                                if !output.blocked_hints.is_empty() {
                                    lines.push("  🚫 Blocked:".to_string());
                                    for hint in &output.blocked_hints {
                                        lines.push(format!(
                                            "    - {} (hint: {})",
                                            hint.rule, hint.suggestion
                                        ));
                                    }
                                }
                            } else {
                                // Fallback: just simplify without metadata
                                let (simplified, _) =
                                    self.core.engine.simplifier.simplify(resolved);
                                if simplified != resolved {
                                    lines.push(format!(
                                        "  Simplified: {}",
                                        DisplayExpr {
                                            context: &self.core.engine.simplifier.context,
                                            id: simplified
                                        }
                                    ));
                                }
                            }
                        }
                        cas_session::EntryKind::Eq { lhs, rhs } => {
                            // Show LHS and RHS
                            lines.push(format!(
                                "  LHS:        {}",
                                DisplayExpr {
                                    context: &self.core.engine.simplifier.context,
                                    id: *lhs
                                }
                            ));
                            lines.push(format!(
                                "  RHS:        {}",
                                DisplayExpr {
                                    context: &self.core.engine.simplifier.context,
                                    id: *rhs
                                }
                            ));

                            // Note about equation-as-expression
                            lines.push(String::new());
                            lines.push(
                                "  Note: When used as expression, this becomes (LHS - RHS)."
                                    .to_string(),
                            );
                        }
                    }
                    reply_output(lines.join("\n"))
                } else {
                    // Check if this ID was ever assigned (it's above next_id means never existed)
                    // Entry not found — could be deleted or never existed
                    reply_output(format!(
                        "Error: Entry #{} not found.\nHint: Use 'history' to see available entries.",
                        id
                    ))
                }
            }
            Err(_) => reply_output("Error: Invalid entry ID. Use 'show #N' or 'show N'."),
        }
    }

    /// Handle "del #id [#id...]" command - delete session entries
    pub(crate) fn handle_del_command(&mut self, line: &str) {
        let reply = self.handle_del_command_core(line);
        self.print_reply(reply);
    }

    fn handle_del_command_core(&mut self, line: &str) -> ReplReply {
        let ids: Vec<u64> = line
            .split_whitespace()
            .filter_map(|s| s.trim_start_matches('#').parse::<u64>().ok())
            .collect();

        if ids.is_empty() {
            return reply_output("Error: No valid IDs specified. Use 'del #1 #2' or 'del 1 2'.");
        }

        let before_len = self.core.state.store().len();
        self.core.state.store_mut().remove(&ids);
        let removed = before_len - self.core.state.store().len();

        if removed > 0 {
            let id_str: Vec<String> = ids.iter().map(|id| format!("#{}", id)).collect();
            reply_output(format!(
                "Deleted {} entry/entries: {}",
                removed,
                id_str.join(", ")
            ))
        } else {
            reply_output("No entries found with the specified IDs.")
        }
    }

    // ========== END SESSION ENVIRONMENT HANDLERS ==========

    pub(crate) fn handle_det(&mut self, line: &str) {
        let reply = self.handle_det_core(line, self.verbosity);
        self.print_reply(reply);
    }

    fn handle_det_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        let rest = line[4..].trim(); // Remove "det "

        // Parse the matrix expression
        match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => {
                // Wrap in det() function call
                let det_expr = self.core.engine.simplifier.context.call("det", vec![expr]);

                // Simplify to compute determinant
                let (result, steps) = self.core.engine.simplifier.simplify(det_expr);

                let mut lines = vec![format!("Parsed: det({})", rest)];

                // Print steps if verbosity is not None
                if verbosity != Verbosity::None && !steps.is_empty() {
                    lines.push("Steps:".to_string());
                    for (i, step) in steps.iter().enumerate() {
                        lines.push(format!(
                            "{}. {}  [{}]",
                            i + 1,
                            step.description,
                            step.rule_name
                        ));
                        for event in step.assumption_events() {
                            if event.kind.should_display() {
                                lines.push(format!(
                                    "   {} {}: {}",
                                    event.kind.icon(),
                                    event.kind.label(),
                                    event.message
                                ));
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
            Err(e) => reply_output(format!("Parse error: {}", e)),
        }
    }

    pub(crate) fn handle_transpose(&mut self, line: &str) {
        let reply = self.handle_transpose_core(line, self.verbosity);
        self.print_reply(reply);
    }

    fn handle_transpose_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        let rest = line[10..].trim(); // Remove "transpose "

        // Parse the matrix expression
        match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => {
                // Wrap in transpose() function call
                let transpose_expr = self
                    .core
                    .engine
                    .simplifier
                    .context
                    .call("transpose", vec![expr]);

                // Simplify to compute transpose
                let (result, steps) = self.core.engine.simplifier.simplify(transpose_expr);

                let mut lines = vec![format!("Parsed: transpose({})", rest)];

                // Print steps if verbosity is not None
                if verbosity != Verbosity::None && !steps.is_empty() {
                    lines.push("Steps:".to_string());
                    for (i, step) in steps.iter().enumerate() {
                        lines.push(format!(
                            "{}. {}  [{}]",
                            i + 1,
                            step.description,
                            step.rule_name
                        ));
                    }
                }

                lines.push(format!(
                    "Result: {}",
                    DisplayExpr {
                        context: &self.core.engine.simplifier.context,
                        id: result
                    }
                ));
                reply_output(lines.join("\n"))
            }
            Err(e) => reply_output(format!("Parse error: {}", e)),
        }
    }

    pub(crate) fn handle_trace(&mut self, line: &str) {
        let reply = self.handle_trace_core(line, self.verbosity);
        self.print_reply(reply);
    }

    fn handle_trace_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        let rest = line[6..].trim(); // Remove "trace "

        // Parse the matrix expression
        match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => {
                // Wrap in trace() function call
                let trace_expr = self
                    .core
                    .engine
                    .simplifier
                    .context
                    .call("trace", vec![expr]);

                // Simplify to compute trace
                let (result, steps) = self.core.engine.simplifier.simplify(trace_expr);

                let mut lines = vec![format!("Parsed: trace({})", rest)];

                // Print steps if verbosity is not None
                if verbosity != Verbosity::None && !steps.is_empty() {
                    lines.push("Steps:".to_string());
                    for (i, step) in steps.iter().enumerate() {
                        lines.push(format!(
                            "{}. {}  [{}]",
                            i + 1,
                            step.description,
                            step.rule_name
                        ));
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
            Err(e) => reply_output(format!("Parse error: {}", e)),
        }
    }

    /// Handle the 'telescope' command for proving telescoping identities like Dirichlet kernel
    pub(crate) fn handle_telescope(&mut self, line: &str) {
        let reply = self.handle_telescope_core(line);
        self.print_reply(reply);
    }

    fn handle_telescope_core(&mut self, line: &str) -> ReplReply {
        let rest = line[10..].trim(); // Remove "telescope "

        if rest.is_empty() {
            return reply_output(
                "Usage: telescope <expression>\n\
                 Example: telescope 1 + 2*cos(x) + 2*cos(2*x) - sin(5*x/2)/sin(x/2)",
            );
        }

        // Parse the expression
        match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => {
                // Apply telescoping strategy
                let result = cas_solver::telescoping::telescope(
                    &mut self.core.engine.simplifier.context,
                    expr,
                );

                // Return formatted output
                reply_output(format!(
                    "Parsed: {}\n\n{}",
                    rest,
                    result.format(&self.core.engine.simplifier.context)
                ))
            }
            Err(e) => reply_output(format!("Parse error: {}", e)),
        }
    }

    /// Handle the 'expand' command for aggressive polynomial expansion
    /// Uses the engine `expand()` path which distributes without educational guards
    pub(crate) fn handle_expand(&mut self, line: &str) {
        let rest = line.strip_prefix("expand").unwrap_or(line).trim();
        if rest.is_empty() {
            self.print_reply(reply_output(
                "Usage: expand <expr>\n\
                 Description: Aggressively expands and distributes polynomials.\n\
                 Example: expand 1/2 * (sqrt(2) - 1) → sqrt(2)/2 - 1/2",
            ));
            return;
        }

        // V2.14.34: Delegate to normal line processing with expand() function wrapper
        // This ensures steps are shown, consistent with using expand() as a function
        let wrapped = format!("expand({})", rest);
        self.handle_eval(&wrapped);
    }

    /// Handle the 'expand_log' command for explicit logarithm expansion
    /// Expands ln(xy) → ln(x) + ln(y), ln(x/y) → ln(x) - ln(y), ln(x^n) → n*ln(x)
    pub(crate) fn handle_expand_log(&mut self, line: &str) {
        let reply = self.handle_expand_log_core(line);
        self.print_reply(reply);
    }

    fn handle_expand_log_core(&mut self, line: &str) -> ReplReply {
        use cas_formatter::DisplayExpr;

        let rest = line.strip_prefix("expand_log").unwrap_or(line).trim();
        if rest.is_empty() {
            return reply_output(
                "Usage: expand_log <expr>\n\
                 Description: Expand logarithms using log properties.\n\
                 Transformations:\n\
                   ln(x*y)   → ln(x) + ln(y)\n\
                   ln(x/y)   → ln(x) - ln(y)\n\
                   ln(x^n)   → n * ln(x)\n\
                 Example: expand_log ln(x^2 * y) → 2*ln(x) + ln(y)",
            );
        }

        match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => {
                let parsed_str = format!(
                    "Parsed: {}",
                    DisplayExpr {
                        context: &self.core.engine.simplifier.context,
                        id: expr
                    }
                );

                // Apply LogExpansionRule recursively to all subexpressions
                let expanded = self.expand_log_recursive(expr);

                // NOTE: We do NOT call simplify() here because LogContractionRule
                // (which is in default rules) would immediately undo the expansion.
                // The expanded form is the desired result.

                let result_str = format!(
                    "Result: {}",
                    clean_display_string(&format!(
                        "{}",
                        DisplayExpr {
                            context: &self.core.engine.simplifier.context,
                            id: expanded
                        }
                    ))
                );
                reply_output(format!("{}\n{}", parsed_str, result_str))
            }
            Err(e) => reply_output(format!("Parse error: {:?}", e)),
        }
    }
}
