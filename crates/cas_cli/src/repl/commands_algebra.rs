impl Repl {
    fn handle_budget_command(&mut self, line: &str) {
        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "budget" - show current setting
                let budget = self.state.options.budget;
                println!("Solve budget: max_branches={}", budget.max_branches);
                println!("  Controls how many case splits the solver can create.");
                println!("  0: No splits (fallback to simple solutions)");
                println!("  1: Conservative (default)");
                println!("  2+: Allow case splits for symbolic bases (a^x=a, etc)");
                println!("  (use 'budget N' to change, e.g. 'budget 2')");
            }
            Some(n_str) => {
                if let Ok(n) = n_str.parse::<usize>() {
                    self.state.options.budget.max_branches = n;
                    println!("Solve budget: max_branches = {}", n);
                    if n == 0 {
                        println!("  ⚠️ No case splits allowed (fallback to simple solutions)");
                    } else if n == 1 {
                        println!("  Conservative mode (default)");
                    } else {
                        println!("  ✓ Case splits enabled for symbolic bases");
                        println!("  Try: solve a^x = a");
                    }
                } else {
                    println!("Invalid budget value: '{}' (expected a number)", n_str);
                    println!("Usage: budget N");
                    println!("  budget 0  - No case splits");
                    println!("  budget 1  - Conservative (default)");
                    println!("  budget 2  - Allow case splits for a^x=a patterns");
                }
            }
        }
    }

    /// Handle "history" or "list" command - show session history
    fn handle_history_command(&self) {
        let entries = self.state.store.list();
        if entries.is_empty() {
            println!("No entries in session history.");
            return;
        }

        println!("Session history ({} entries):", entries.len());
        for entry in entries {
            let type_indicator = match &entry.kind {
                cas_engine::EntryKind::Expr(_) => "Expr",
                cas_engine::EntryKind::Eq { .. } => "Eq  ",
            };
            // Show simplified form if possible
            let display = match &entry.kind {
                cas_engine::EntryKind::Expr(expr_id) => {
                    format!(
                        "{}",
                        cas_ast::DisplayExpr {
                            context: &self.engine.simplifier.context,
                            id: *expr_id
                        }
                    )
                }
                cas_engine::EntryKind::Eq { lhs, rhs } => {
                    format!(
                        "{} = {}",
                        cas_ast::DisplayExpr {
                            context: &self.engine.simplifier.context,
                            id: *lhs
                        },
                        cas_ast::DisplayExpr {
                            context: &self.engine.simplifier.context,
                            id: *rhs
                        }
                    )
                }
            };
            println!("  #{:<3} [{}] {}", entry.id, type_indicator, display);
        }
    }

    /// Handle "show #id" command - show details of a specific entry
    fn handle_show_command(&mut self, line: &str) {
        let input = line.trim().trim_start_matches('#');
        match input.parse::<u64>() {
            Ok(id) => {
                // Clone entry data to avoid borrow conflicts
                let entry_data = self
                    .state
                    .store
                    .get(id)
                    .map(|e| (e.type_str().to_string(), e.raw_text.clone(), e.kind.clone()));

                if let Some((type_str, raw_text, kind)) = entry_data {
                    println!("Entry #{}:", id);
                    println!("  Type:       {}", type_str);
                    println!("  Raw:        {}", raw_text);

                    match &kind {
                        cas_engine::EntryKind::Expr(expr_id) => {
                            // Show parsed expression
                            println!(
                                "  Parsed:     {}",
                                DisplayExpr {
                                    context: &self.engine.simplifier.context,
                                    id: *expr_id
                                }
                            );

                            // Show resolved (after #id and env substitution)
                            let resolved = match self
                                .state
                                .resolve_all(&mut self.engine.simplifier.context, *expr_id)
                            {
                                Ok(r) => r,
                                Err(_) => *expr_id,
                            };
                            if resolved != *expr_id {
                                println!(
                                    "  Resolved:   {}",
                                    DisplayExpr {
                                        context: &self.engine.simplifier.context,
                                        id: resolved
                                    }
                                );
                            }

                            // Perform full eval to get requires/assumed metadata
                            let req = cas_engine::EvalRequest {
                                raw_input: raw_text.clone(),
                                parsed: *expr_id,
                                kind: cas_engine::EntryKind::Expr(*expr_id),
                                action: cas_engine::EvalAction::Simplify,
                                auto_store: false,
                            };

                            if let Ok(output) = self.engine.eval(&mut self.state, req) {
                                // Show simplified result
                                if let cas_engine::EvalResult::Expr(simplified) = &output.result {
                                    if *simplified != *expr_id {
                                        println!(
                                            "  Simplified: {}",
                                            DisplayExpr {
                                                context: &self.engine.simplifier.context,
                                                id: *simplified
                                            }
                                        );
                                    }
                                }

                                // Show Requires (implicit domain conditions)
                                if !output.required_conditions.is_empty() {
                                    println!("  ℹ️ Requires:");
                                    for cond in &output.required_conditions {
                                        println!(
                                            "    - {}",
                                            cond.display(&self.engine.simplifier.context)
                                        );
                                    }
                                }

                                // Show Assumed (domain warnings)
                                if !output.domain_warnings.is_empty() {
                                    println!("  ⚠ Assumed:");
                                    for w in &output.domain_warnings {
                                        println!("    - {}", w.message);
                                    }
                                }

                                // Show Blocked hints (rules that couldn't fire)
                                if !output.blocked_hints.is_empty() {
                                    println!("  🚫 Blocked:");
                                    for hint in &output.blocked_hints {
                                        println!("    - {} (hint: {})", hint.rule, hint.suggestion);
                                    }
                                }
                            } else {
                                // Fallback: just simplify without metadata
                                let (simplified, _) = self.engine.simplifier.simplify(resolved);
                                if simplified != resolved {
                                    println!(
                                        "  Simplified: {}",
                                        DisplayExpr {
                                            context: &self.engine.simplifier.context,
                                            id: simplified
                                        }
                                    );
                                }
                            }
                        }
                        cas_engine::EntryKind::Eq { lhs, rhs } => {
                            // Show LHS and RHS
                            println!(
                                "  LHS:        {}",
                                DisplayExpr {
                                    context: &self.engine.simplifier.context,
                                    id: *lhs
                                }
                            );
                            println!(
                                "  RHS:        {}",
                                DisplayExpr {
                                    context: &self.engine.simplifier.context,
                                    id: *rhs
                                }
                            );

                            // Note about equation-as-expression
                            println!();
                            println!("  Note: When used as expression, this becomes (LHS - RHS).");
                        }
                    }
                } else {
                    // Check if this ID was ever assigned (it's above next_id means never existed)
                    // Entry not found — could be deleted or never existed
                    println!("Error: Entry #{} not found.", id);
                    println!("Hint: Use 'history' to see available entries.");
                }
            }
            Err(_) => {
                println!("Error: Invalid entry ID. Use 'show #N' or 'show N'.");
            }
        }
    }

    /// Handle "del #id [#id...]" command - delete session entries
    fn handle_del_command(&mut self, line: &str) {
        let ids: Vec<u64> = line
            .split_whitespace()
            .filter_map(|s| s.trim_start_matches('#').parse::<u64>().ok())
            .collect();

        if ids.is_empty() {
            println!("Error: No valid IDs specified. Use 'del #1 #2' or 'del 1 2'.");
            return;
        }

        let before_len = self.state.store.len();
        self.state.store.remove(&ids);
        let removed = before_len - self.state.store.len();

        if removed > 0 {
            let id_str: Vec<String> = ids.iter().map(|id| format!("#{}", id)).collect();
            println!("Deleted {} entry/entries: {}", removed, id_str.join(", "));
        } else {
            println!("No entries found with the specified IDs.");
        }
    }

    // ========== END SESSION ENVIRONMENT HANDLERS ==========

    fn handle_det(&mut self, line: &str) {
        let rest = line[4..].trim(); // Remove "det "

        // Parse the matrix expression
        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                // Wrap in det() function call
                let det_expr = self
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Function("det".to_string(), vec![expr]));

                // Simplify to compute determinant
                let (result, steps) = self.engine.simplifier.simplify(det_expr);

                println!("Parsed: det({})", rest);

                // Print steps if verbosity is not None
                if self.verbosity != Verbosity::None && !steps.is_empty() {
                    println!("Steps:");
                    for (i, step) in steps.iter().enumerate() {
                        println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
                        for event in &step.assumption_events {
                            if event.kind.should_display() {
                                println!(
                                    "   {} {}: {}",
                                    event.kind.icon(),
                                    event.kind.label(),
                                    event.message
                                );
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
            Err(e) => println!("Parse error: {}", e),
        }
    }

    fn handle_transpose(&mut self, line: &str) {
        let rest = line[10..].trim(); // Remove "transpose "

        // Parse the matrix expression
        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                // Wrap in transpose() function call
                let transpose_expr = self
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Function("transpose".to_string(), vec![expr]));

                // Simplify to compute transpose
                let (result, steps) = self.engine.simplifier.simplify(transpose_expr);

                println!("Parsed: transpose({})", rest);

                // Print steps if verbosity is not None
                if self.verbosity != Verbosity::None && !steps.is_empty() {
                    println!("Steps:");
                    for (i, step) in steps.iter().enumerate() {
                        println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
                    }
                }

                println!(
                    "Result: {}",
                    DisplayExpr {
                        context: &self.engine.simplifier.context,
                        id: result
                    }
                );
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }

    fn handle_trace(&mut self, line: &str) {
        let rest = line[6..].trim(); // Remove "trace "

        // Parse the matrix expression
        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                // Wrap in trace() function call
                let trace_expr = self
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Function("trace".to_string(), vec![expr]));

                // Simplify to compute trace
                let (result, steps) = self.engine.simplifier.simplify(trace_expr);

                println!("Parsed: trace({})", rest);

                // Print steps if verbosity is not None
                if self.verbosity != Verbosity::None && !steps.is_empty() {
                    println!("Steps:");
                    for (i, step) in steps.iter().enumerate() {
                        println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
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
            Err(e) => println!("Parse error: {}", e),
        }
    }

    /// Handle the 'telescope' command for proving telescoping identities like Dirichlet kernel
    fn handle_telescope(&mut self, line: &str) {
        let rest = line[10..].trim(); // Remove "telescope "

        if rest.is_empty() {
            println!("Usage: telescope <expression>");
            println!("Example: telescope 1 + 2*cos(x) + 2*cos(2*x) - sin(5*x/2)/sin(x/2)");
            return;
        }

        // Parse the expression
        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                println!("Parsed: {}", rest);
                println!();

                // Apply telescoping strategy
                let result =
                    cas_engine::telescoping::telescope(&mut self.engine.simplifier.context, expr);

                // Print formatted output
                println!("{}", result.format(&self.engine.simplifier.context));
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }

    /// Handle the 'expand' command for aggressive polynomial expansion
    /// Uses cas_engine::expand::expand() which distributes without educational guards
    fn handle_expand(&mut self, line: &str) {
        let rest = line.strip_prefix("expand").unwrap_or(line).trim();
        if rest.is_empty() {
            println!("Usage: expand <expr>");
            println!("Description: Aggressively expands and distributes polynomials.");
            println!("Example: expand 1/2 * (sqrt(2) - 1) → sqrt(2)/2 - 1/2");
            return;
        }

        // V2.14.34: Delegate to normal line processing with expand() function wrapper
        // This ensures steps are shown, consistent with using expand() as a function
        let wrapped = format!("expand({})", rest);
        self.handle_eval(&wrapped);
    }

    /// Handle the 'expand_log' command for explicit logarithm expansion
    /// Expands ln(xy) → ln(x) + ln(y), ln(x/y) → ln(x) - ln(y), ln(x^n) → n*ln(x)
    fn handle_expand_log(&mut self, line: &str) {
        use cas_ast::DisplayExpr;

        let rest = line.strip_prefix("expand_log").unwrap_or(line).trim();
        if rest.is_empty() {
            println!("Usage: expand_log <expr>");
            println!("Description: Expand logarithms using log properties.");
            println!("Transformations:");
            println!("  ln(x*y)   → ln(x) + ln(y)");
            println!("  ln(x/y)   → ln(x) - ln(y)");
            println!("  ln(x^n)   → n * ln(x)");
            println!("Example: expand_log ln(x^2 * y) → 2*ln(x) + ln(y)");
            return;
        }

        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                println!(
                    "Parsed: {}",
                    DisplayExpr {
                        context: &self.engine.simplifier.context,
                        id: expr
                    }
                );

                // Apply LogExpansionRule recursively to all subexpressions
                let expanded = self.expand_log_recursive(expr);

                // NOTE: We do NOT call simplify() here because LogContractionRule
                // (which is in default rules) would immediately undo the expansion.
                // The expanded form is the desired result.

                println!(
                    "Result: {}",
                    clean_display_string(&format!(
                        "{}",
                        DisplayExpr {
                            context: &self.engine.simplifier.context,
                            id: expanded
                        }
                    ))
                );
            }
            Err(e) => println!("Parse error: {:?}", e),
        }
    }
}
