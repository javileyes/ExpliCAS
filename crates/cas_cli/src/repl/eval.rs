use super::*;

impl Repl {
    /// Legacy wrapper - calls core and prints
    pub(crate) fn handle_eval(&mut self, line: &str) {
        let reply = self.handle_eval_core(line);
        self.print_reply(reply);
    }

    /// Core: handle eval command, returns ReplReply (no I/O)
    pub(crate) fn handle_eval_core(&mut self, line: &str) -> ReplReply {
        use cas_formatter::root_style::ParseStyleSignals;

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
                        for line in cas_solver::format_domain_warning_lines(
                            &output.domain_warnings,
                            true,
                            "⚠ ",
                        ) {
                            reply.push(ReplMsg::warn(line));
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

                            let display_level = self.core.state.options().requires_display;
                            let debug_mode = self.core.debug_mode;
                            let requires_lines = cas_solver::format_diagnostics_requires_lines(
                                &mut self.core.engine.simplifier.context,
                                &output.diagnostics,
                                result_expr,
                                display_level,
                                debug_mode,
                            );

                            if !requires_lines.is_empty() {
                                reply.push(ReplMsg::info("ℹ️ Requires:".to_string()));
                                for line in requires_lines {
                                    reply.push(ReplMsg::info(line));
                                }
                            }
                        }

                        // Collect assumptions from steps for assumption reporting (before steps are consumed)
                        let show_assumptions =
                            self.core.state.options().shared.assumption_reporting
                                != cas_solver::AssumptionReporting::Off;
                        let assumed_conditions: Vec<(String, String)> = if show_assumptions {
                            cas_solver::collect_assumed_conditions_from_steps(&output.steps)
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
                        let style_prefs = cas_formatter::root_style::StylePreferences::from_expression_with_signals(
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
                                    cas_solver::display_solution_set(ctx, solution_set)
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
                        let hints = cas_solver::filter_blocked_hints_for_eval(
                            &self.core.engine.simplifier.context,
                            output.resolved,
                            &output.blocked_hints,
                        );
                        if !hints.is_empty() && self.core.state.options().hints_enabled {
                            let domain_mode =
                                self.core.state.options().shared.semantics.domain_mode;
                            for line in cas_solver::format_eval_blocked_hints_lines(
                                &self.core.engine.simplifier.context,
                                &hints,
                                domain_mode,
                            ) {
                                reply.push(ReplMsg::info(line));
                            }
                        }

                        // Show assumptions summary when assumption_reporting is enabled (after hints)
                        if show_assumptions && !assumed_conditions.is_empty() {
                            for line in cas_solver::format_assumed_conditions_report_lines(
                                &assumed_conditions,
                            ) {
                                reply.push(ReplMsg::info(line));
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
