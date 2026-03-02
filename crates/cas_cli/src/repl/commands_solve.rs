use super::*;

impl Repl {
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

        match cas_solver::evaluate_weierstrass_input(&mut self.core.engine.simplifier, rest) {
            Ok(out) => {
                use cas_formatter::DisplayExpr;
                let result_str = format!(
                    "{}",
                    DisplayExpr {
                        context: &self.core.engine.simplifier.context,
                        id: out.substituted_expr
                    }
                );
                let simplified_str = clean_display_string(&format!(
                    "{}",
                    DisplayExpr {
                        context: &self.core.engine.simplifier.context,
                        id: out.simplified_expr
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
            Err(cas_solver::TransformEvalError::Parse(e)) => {
                reply_output(format!("Parse error: {}", e))
            }
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

        let parsed_input = cas_solver::parse_solve_command_input(rest);
        let eq_str = parsed_input.equation.trim();
        let var_explicit = parsed_input.variable;

        let prepared = {
            let ctx = &mut self.core.engine.simplifier.context;
            cas_solver::prepare_timeline_solve_input(ctx, eq_str, var_explicit)
        };

        match prepared {
            Ok(prepared) => {
                let eq = prepared.equation;
                let var = prepared.var;

                // Call solver with step collection enabled and semantic options
                self.core.engine.simplifier.set_collect_steps(true);
                let solver_opts = cas_solver::SolverOptions {
                    value_domain: self.core.state.options().shared.semantics.value_domain,
                    domain_mode: self.core.state.options().shared.semantics.domain_mode,
                    assume_scope: self.core.state.options().shared.semantics.assume_scope,
                    budget: self.core.state.options().budget,
                    ..Default::default()
                };

                // V2.9.8: Use type-safe API that includes automatic cleanup
                match cas_solver::solve_with_display_steps(
                    &eq,
                    &var,
                    &mut self.core.engine.simplifier,
                    solver_opts,
                ) {
                    Ok((solution_set, display_steps, _diagnostics)) => {
                        if display_steps.0.is_empty() {
                            let result_str = cas_solver::display_solution_set(
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
                        let mut timeline = cas_didactic::SolveTimelineHtml::new(
                            &mut self.core.engine.simplifier.context,
                            &display_steps.0,
                            &eq,
                            &solution_set,
                            &var,
                        );
                        let html = timeline.to_html();

                        // Return WriteFile action + result
                        let result_str = cas_solver::display_solution_set(
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
            Err(cas_solver::SolvePrepareError::ExpectedEquation) => reply_output(
                "Error: Expected an equation for solve timeline, got an expression.\n\
                     Usage: timeline solve <equation>, <variable>\n\
                     Example: timeline solve x + 2 = 5, x",
            ),
            Err(cas_solver::SolvePrepareError::ParseError(e)) => {
                reply_output(format!("Error parsing equation: {}", e))
            }
            Err(cas_solver::SolvePrepareError::NoVariable) => reply_output(
                "Error: timeline solve found no variable.\n\
                 Use timeline solve <equation>, <variable>",
            ),
            Err(cas_solver::SolvePrepareError::AmbiguousVariables(vars)) => reply_output(format!(
                "Error: timeline solve found ambiguous variables {{{}}}.\n\
                 Use timeline solve <equation>, {}",
                vars.join(", "),
                vars.first().unwrap_or(&"x".to_string())
            )),
        }
    }

    pub(crate) fn handle_solve(&mut self, line: &str) {
        let reply = self.handle_solve_core(line, self.verbosity);
        self.print_reply(reply);
    }

    fn handle_solve_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        use cas_solver::EvalResult;

        let mut lines: Vec<String> = Vec::new();

        // solve [--check] <equation>, <var>
        let rest = line[6..].trim();

        // Parse --check flag (one-shot override)
        let (check_enabled, rest) = if let Some(stripped) = rest.strip_prefix("--check") {
            let after_flag = stripped.trim_start();
            (true, after_flag)
        } else {
            // Use session toggle if no explicit flag
            (self.core.state.options().check_solutions, rest)
        };

        let parsed_input = cas_solver::parse_solve_command_input(rest);
        let eq_str = parsed_input.equation.trim();
        let var_explicit = parsed_input.variable;

        let prepared = {
            let ctx = &mut self.core.engine.simplifier.context;
            cas_solver::prepare_solve_eval_request(ctx, eq_str, var_explicit, true)
        };

        match prepared {
            Ok(prepared) => {
                let original_equation = prepared.original_equation;
                let var = prepared.var;
                let req = prepared.request;

                match self.core.engine.eval(&mut self.core.state, req) {
                    Ok(output) => {
                        // Show ID
                        let id_prefix = if let Some(id) = output.stored_id {
                            format!("#{}: ", id)
                        } else {
                            String::new()
                        };
                        lines.push(format!("{}Solving for {}...", id_prefix, var));

                        for line in cas_solver::format_domain_warning_lines(
                            &output.domain_warnings,
                            true,
                            "⚠ ",
                        ) {
                            lines.push(line);
                        }
                        let solver_assumption_records =
                            cas_solver::assumption_records_from_engine(&output.solver_assumptions);

                        // Show solver assumptions summary if any
                        if let Some(summary) = cas_solver::format_assumption_records_summary(
                            &solver_assumption_records,
                        ) {
                            lines.push(format!("⚠ Assumptions: {}", summary));
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
                            lines.extend(cas_solver::format_solve_steps_lines(
                                &self.core.engine.simplifier.context,
                                &output.solve_steps,
                                &output.output_scopes,
                                verbosity == Verbosity::Verbose,
                            ));
                        }

                        match output.result {
                            EvalResult::SolutionSet(ref solution_set) => {
                                // V2.0: Display full solution set including Conditional
                                let ctx = &self.core.engine.simplifier.context;
                                lines.push(format!(
                                    "Result: {}",
                                    cas_solver::display_solution_set(ctx, solution_set)
                                ));
                            }
                            EvalResult::Set(ref sols) => {
                                // Legacy: discrete solutions as Vec<ExprId>
                                let ctx = &self.core.engine.simplifier.context;
                                let sol_strs: Vec<String> = {
                                    let registry = cas_formatter::display_transforms::DisplayTransformRegistry::with_defaults();
                                    let style = StylePreferences::default();
                                    let renderer =
                                        cas_formatter::display_transforms::ScopedRenderer::new(
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
                        let display_level = self.core.state.options().requires_display;
                        let requires_lines = cas_solver::format_diagnostics_requires_lines(
                            &mut self.core.engine.simplifier.context,
                            &output.diagnostics,
                            Some(result_expr_id),
                            display_level,
                            self.core.debug_mode,
                        );

                        if !requires_lines.is_empty() {
                            lines.push("ℹ️ Requires:".to_string());
                            for line in requires_lines {
                                lines.push(line);
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
                                    let verify_result = cas_solver::verify_solution_set(
                                        &mut self.core.engine.simplifier,
                                        eq,
                                        &var,
                                        solution_set,
                                    );
                                    lines.extend(cas_solver::format_verify_summary_lines(
                                        &self.core.engine.simplifier.context,
                                        &var,
                                        &verify_result,
                                        "  ",
                                    ));
                                }
                            }
                        }

                        // V2.1 Issue #3: Explain mode - structured summary for solve
                        // Collect blocked hints for debug output
                        let hints = cas_solver::take_blocked_hints();
                        let has_assumptions = !solver_assumption_records.is_empty();
                        let has_blocked = !hints.is_empty();

                        if self.core.debug_mode && (has_assumptions || has_blocked) {
                            lines.push(String::new()); // Separator line
                            let ctx = &self.core.engine.simplifier.context;
                            let domain_mode =
                                self.core.state.options().shared.semantics.domain_mode;

                            // Block 1: Assumptions used
                            if has_assumptions {
                                for line in cas_solver::format_assumption_records_section_lines(
                                    &solver_assumption_records,
                                    "ℹ️ Assumptions used:",
                                    "  - ",
                                ) {
                                    lines.push(line);
                                }
                            }

                            // Block 2: Blocked simplifications
                            if has_blocked {
                                for line in cas_solver::format_blocked_simplifications_section_lines(
                                    ctx,
                                    &hints,
                                    domain_mode,
                                ) {
                                    lines.push(line);
                                }
                            }
                        } else if has_blocked && self.core.state.options().hints_enabled {
                            // Legacy: show blocked hints even without debug_mode if hints_enabled
                            let ctx = &self.core.engine.simplifier.context;
                            let domain_mode =
                                self.core.state.options().shared.semantics.domain_mode;

                            lines.push(String::new());
                            for line in cas_solver::format_blocked_simplifications_section_lines(
                                ctx,
                                &hints,
                                domain_mode,
                            ) {
                                lines.push(line);
                            }
                        }
                    }
                    Err(e) => lines.push(format!("Error: {}", e)),
                }
            }
            Err(cas_solver::SolvePrepareError::ParseError(e)) => {
                lines.push(format!("Parse error: {}", e))
            }
            Err(cas_solver::SolvePrepareError::NoVariable) => {
                lines.push(
                    "Error: solve() found no variable to solve for.\n\
                     Use solve(expr, x) to specify the variable."
                        .to_string(),
                );
            }
            Err(cas_solver::SolvePrepareError::AmbiguousVariables(vars)) => {
                lines.push(format!(
                    "Error: solve() found ambiguous variables {{{}}}.\n\
                     Use solve(expr, {}) or solve(expr, {{{}}}).",
                    vars.join(", "),
                    vars.first().unwrap_or(&"x".to_string()),
                    vars.join(", ")
                ));
            }
            Err(cas_solver::SolvePrepareError::ExpectedEquation) => {
                lines.push("Parse error: expected equation".to_string());
            }
        }

        reply_output(lines.join("\n"))
    }
}
