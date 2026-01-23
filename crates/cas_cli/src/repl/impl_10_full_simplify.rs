impl Repl {
    fn handle_full_simplify(&mut self, line: &str) {
        // simplify <expr>
        // Uses a temporary simplifier with ALL default rules (including aggressive distribution)
        let expr_str = line[9..].trim();

        // We need to use the existing context to parse, but then we want to simplify using a different rule set.
        // The Simplifier struct owns the context.
        // Option 1: Create a new Simplifier, parse into it.
        // Option 2: Swap rules in current simplifier? (Hard)
        // Option 3: Create a new Simplifier, copy context? (Hard)

        // Easiest: Create new simplifier, parse string into it.
        // Note: Variables from previous history won't be available if we don't copy context.
        // But REPL history is just text in rustyline, not context state (unless we implement variable storage).
        // Current implementation: Context is reset per line? No, self.engine.simplifier.context persists.
        // If we want to support "x = 5; simplify x", we need to share context.

        // Better approach:
        // 1. Parse expression using current context.
        // 2. Create a temporary Simplifier that SHARES the context?
        //    Simplifier owns Context. We can't easily share.
        //    But we can temporarily TAKE the context, use it in a new Simplifier, and then put it back.

        let mut temp_simplifier = Simplifier::with_default_rules();
        // Swap context and profiler so temp_simplifier uses main profiler
        std::mem::swap(
            &mut self.engine.simplifier.context,
            &mut temp_simplifier.context,
        );
        std::mem::swap(
            &mut self.engine.simplifier.profiler,
            &mut temp_simplifier.profiler,
        );

        // Ensure we have the aggressive rules we want (DistributeRule is in default)
        // Also add DistributeConstantRule just in case (though DistributeRule covers it)

        // Set steps mode
        temp_simplifier.set_collect_steps(self.verbosity != Verbosity::None);

        match cas_parser::parse(expr_str, &mut temp_simplifier.context) {
            Ok(expr) => {
                // Note: Tool dispatcher is handled in Engine::eval, not here
                // This code path is for timeline/specific commands, not regular expression evaluation

                // Resolve session variables (A, B, etc.) before simplifying
                let resolved_expr = match self.state.resolve_all(&mut temp_simplifier.context, expr)
                {
                    Ok(resolved) => resolved,
                    Err(e) => {
                        println!("Error resolving variables: {:?}", e);
                        // Swap context and profiler back before returning
                        std::mem::swap(
                            &mut self.engine.simplifier.context,
                            &mut temp_simplifier.context,
                        );
                        std::mem::swap(
                            &mut self.engine.simplifier.profiler,
                            &mut temp_simplifier.profiler,
                        );
                        return;
                    }
                };

                // STYLE SNIFFING: Detect user's preferred notation BEFORE processing
                // Parse equation part
                // Style signals handled during display logic mostly, removing invalid context access
                let style_signals = ParseStyleSignals::from_input_string(expr_str);
                let style_prefs = StylePreferences::from_expression_with_signals(
                    &temp_simplifier.context,
                    resolved_expr,
                    Some(&style_signals),
                );

                println!(
                    "Parsed: {}",
                    DisplayExpr {
                        context: &temp_simplifier.context,
                        id: resolved_expr
                    }
                );
                // Use session options (expand_policy, context_mode, etc.) for simplification
                let mut opts = self.state.options.to_simplify_options();
                opts.collect_steps = self.verbosity != Verbosity::None;

                let (simplified, steps, _stats) =
                    temp_simplifier.simplify_with_stats(resolved_expr, opts);

                if self.verbosity != Verbosity::None {
                    if steps.is_empty() {
                        if self.verbosity != Verbosity::Succinct {
                            println!("No simplification steps needed.");
                        }
                    } else {
                        if self.verbosity != Verbosity::Succinct {
                            println!("Steps (Aggressive Mode):");
                        }
                        let mut current_root = expr;
                        let mut step_count = 0;
                        for step in steps.iter() {
                            if should_show_step(step, self.verbosity) {
                                step_count += 1;

                                if self.verbosity == Verbosity::Succinct {
                                    // Low mode: just global state
                                    current_root = reconstruct_global_expr(
                                        &mut temp_simplifier.context,
                                        current_root,
                                        &step.path,
                                        step.after,
                                    );
                                    println!(
                                        "-> {}",
                                        DisplayExpr {
                                            context: &temp_simplifier.context,
                                            id: current_root
                                        }
                                    );
                                } else {
                                    // Normal/Verbose
                                    println!(
                                        "{}. {}  [{}]",
                                        step_count, step.description, step.rule_name
                                    );

                                    if self.verbosity == Verbosity::Verbose
                                        || self.verbosity == Verbosity::Normal
                                    {
                                        // Show Before: global expression before this step
                                        if let Some(global_before) = step.global_before {
                                            println!(
                                                "   Before: {}",
                                                clean_display_string(&format!(
                                                    "{}",
                                                    DisplayExprStyled::new(
                                                        &temp_simplifier.context,
                                                        global_before,
                                                        &style_prefs
                                                    )
                                                ))
                                            );
                                        } else {
                                            println!(
                                                "   Before: {}",
                                                clean_display_string(&format!(
                                                    "{}",
                                                    DisplayExprStyled::new(
                                                        &temp_simplifier.context,
                                                        current_root,
                                                        &style_prefs
                                                    )
                                                ))
                                            );
                                        }

                                        // Show Rule: local transformation
                                        // Use before_local/after_local if available (for n-ary rules),
                                        // otherwise fall back to before/after
                                        let (rule_before_id, rule_after_id) =
                                            match (step.before_local, step.after_local) {
                                                (Some(bl), Some(al)) => (bl, al),
                                                _ => (step.before, step.after),
                                            };

                                        let before_disp = clean_display_string(&format!(
                                            "{}",
                                            DisplayExprStyled::new(
                                                &temp_simplifier.context,
                                                rule_before_id,
                                                &style_prefs
                                            )
                                        ));
                                        // Use scoped renderer for after expression if rule has transforms
                                        let after_disp =
                                            clean_display_string(&render_with_rule_scope(
                                                &temp_simplifier.context,
                                                rule_after_id,
                                                &step.rule_name,
                                                &style_prefs,
                                            ));

                                        println!("   Rule: {} -> {}", before_disp, after_disp);
                                    }

                                    // Use precomputed global_after if available, fall back to reconstruction
                                    if let Some(global_after) = step.global_after {
                                        current_root = global_after;
                                    } else {
                                        current_root = reconstruct_global_expr(
                                            &mut temp_simplifier.context,
                                            current_root,
                                            &step.path,
                                            step.after,
                                        );
                                    }

                                    // Show After: global expression after this step
                                    println!(
                                        "   After: {}",
                                        clean_display_string(&format!(
                                            "{}",
                                            DisplayExprStyled::new(
                                                &temp_simplifier.context,
                                                current_root,
                                                &style_prefs
                                            )
                                        ))
                                    );

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
                            } else {
                                // Step not shown, but still update current_root for subsequent steps
                                if let Some(global_after) = step.global_after {
                                    current_root = global_after;
                                } else {
                                    current_root = reconstruct_global_expr(
                                        &mut temp_simplifier.context,
                                        current_root,
                                        &step.path,
                                        step.after,
                                    );
                                }
                            }
                        }
                    }
                }
                // Use DisplayExprStyled with detected preferences for consistent output
                println!(
                    "Result: {}",
                    clean_display_string(&format!(
                        "{}",
                        DisplayExprStyled::new(&temp_simplifier.context, simplified, &style_prefs)
                    ))
                );
            }
            Err(e) => println!("Error: {}", e),
        }

        // Swap context and profiler back
        std::mem::swap(
            &mut self.engine.simplifier.context,
            &mut temp_simplifier.context,
        );
        std::mem::swap(
            &mut self.engine.simplifier.profiler,
            &mut temp_simplifier.profiler,
        );

        // Store health report for the `health` command (if health tracking is enabled)
        if self.health_enabled {
            self.last_health_report = Some(self.engine.simplifier.profiler.health_report());
        }
    }
}
