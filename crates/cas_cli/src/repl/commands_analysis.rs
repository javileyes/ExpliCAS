use super::*;

fn extract_equiv_command_tail(line: &str) -> &str {
    line.strip_prefix("equiv").unwrap_or(line).trim()
}

fn extract_substitute_command_tail(line: &str) -> &str {
    line.strip_prefix("subst").unwrap_or(line).trim()
}

fn extract_explain_command_tail(line: &str) -> &str {
    line.strip_prefix("explain").unwrap_or(line).trim()
}

fn extract_timeline_command_tail(line: &str) -> &str {
    line.strip_prefix("timeline").unwrap_or(line).trim()
}

fn extract_visualize_command_tail(line: &str) -> &str {
    line.strip_prefix("visualize ")
        .or_else(|| line.strip_prefix("viz "))
        .unwrap_or(line)
        .trim()
}

fn visualize_output_hint_lines() -> [&'static str; 2] {
    [
        "Render with: dot -Tsvg ast.dot -o ast.svg",
        "Or: dot -Tpng ast.dot -o ast.png",
    ]
}

fn split_by_comma_ignoring_parens(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut balance = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' | '{' => balance += 1,
            ')' | ']' | '}' => balance -= 1,
            ',' if balance == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    parts.push(&s[start..]);
    parts
}

fn rsplit_ignoring_parens(s: &str, delimiter: char) -> Option<(&str, &str)> {
    let mut balance = 0;
    let mut split_idx = None;

    for (i, c) in s.char_indices().rev() {
        if c == ')' {
            balance += 1;
        } else if c == '(' {
            balance -= 1;
        } else if c == delimiter && balance == 0 {
            split_idx = Some(i);
            break;
        }
    }

    split_idx.map(|idx| (&s[..idx], &s[idx + 1..]))
}

fn parse_statement_or_session_ref(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<cas_parser::Statement, String> {
    if input.starts_with('#') && input[1..].chars().all(char::is_numeric) {
        Ok(cas_parser::Statement::Expression(ctx.var(input)))
    } else {
        cas_parser::parse_statement(input, ctx).map_err(|e| e.to_string())
    }
}

fn statement_to_expr_id(
    ctx: &mut cas_ast::Context,
    stmt: cas_parser::Statement,
) -> cas_ast::ExprId {
    match stmt {
        cas_parser::Statement::Equation(eq) => ctx.call("Equal", vec![eq.lhs, eq.rhs]),
        cas_parser::Statement::Expression(expr) => expr,
    }
}

fn parse_expr_or_equation_as_expr(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<cas_ast::ExprId, String> {
    let stmt = parse_statement_or_session_ref(ctx, input)?;
    Ok(statement_to_expr_id(ctx, stmt))
}

fn parse_expr_pair(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<(cas_ast::ExprId, cas_ast::ExprId), super::error_render::ParseExprPairError> {
    let (left, right) = rsplit_ignoring_parens(input, ',')
        .ok_or(super::error_render::ParseExprPairError::MissingDelimiter)?;
    let left = left.trim();
    let right = right.trim();

    let first = parse_expr_or_equation_as_expr(ctx, left)
        .map_err(super::error_render::ParseExprPairError::FirstArg)?;
    let second = parse_expr_or_equation_as_expr(ctx, right)
        .map_err(super::error_render::ParseExprPairError::SecondArg)?;
    Ok((first, second))
}

fn evaluate_equiv_input(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
) -> Result<cas_solver::EquivalenceResult, super::error_render::ParseExprPairError> {
    let (lhs, rhs) = parse_expr_pair(&mut simplifier.context, input)?;
    Ok(simplifier.are_equivalent_extended(lhs, rhs))
}

fn parse_substitute_args(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<
    (cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId),
    super::error_render::ParseSubstituteArgsError,
> {
    let parts = split_by_comma_ignoring_parens(input);
    if parts.len() != 3 {
        return Err(super::error_render::ParseSubstituteArgsError::InvalidArity);
    }

    let expr_str = parts[0].trim();
    let target_str = parts[1].trim();
    let replacement_str = parts[2].trim();

    let expr = cas_parser::parse(expr_str, ctx)
        .map_err(|e| super::error_render::ParseSubstituteArgsError::Expression(e.to_string()))?;
    let target = cas_parser::parse(target_str, ctx)
        .map_err(|e| super::error_render::ParseSubstituteArgsError::Target(e.to_string()))?;
    let replacement = cas_parser::parse(replacement_str, ctx)
        .map_err(|e| super::error_render::ParseSubstituteArgsError::Replacement(e.to_string()))?;

    Ok((expr, target, replacement))
}

fn evaluate_substitute_input(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
    options: cas_solver::SubstituteOptions,
) -> Result<
    (cas_ast::ExprId, cas_solver::SubstituteStrategy),
    super::error_render::ParseSubstituteArgsError,
> {
    let (expr, target, replacement) = parse_substitute_args(&mut simplifier.context, input)?;
    Ok(cas_solver::substitute_auto_with_strategy(
        &mut simplifier.context,
        expr,
        target,
        replacement,
        options,
    ))
}

fn evaluate_substitute_and_simplify_input(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
    options: cas_solver::SubstituteOptions,
) -> Result<SubstituteEvalOutput, super::error_render::ParseSubstituteArgsError> {
    let (substituted_expr, strategy) = evaluate_substitute_input(simplifier, input, options)?;
    let (simplified_expr, steps) = simplifier.simplify(substituted_expr);
    Ok(SubstituteEvalOutput {
        simplified_expr,
        strategy,
        steps,
    })
}

#[derive(Debug, Clone)]
struct SubstituteEvalOutput {
    simplified_expr: cas_ast::ExprId,
    strategy: cas_solver::SubstituteStrategy,
    steps: Vec<cas_solver::Step>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TimelineEvalError {
    Parse(String),
    Eval(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TimelineCommandEvalError {
    Solve(cas_solver::TimelineSolveEvalError),
    Simplify(TimelineEvalError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TimelineCommandInput {
    Solve(String),
    Simplify { expr: String, aggressive: bool },
}

#[derive(Debug, Clone)]
struct TimelineSimplifyEvalOutput {
    parsed_expr: cas_ast::ExprId,
    simplified_expr: cas_ast::ExprId,
    steps: cas_solver::DisplayEvalSteps,
}

fn parse_timeline_command_input(rest: &str) -> TimelineCommandInput {
    if let Some(solve_rest) = rest.strip_prefix("solve ") {
        return TimelineCommandInput::Solve(solve_rest.trim().to_string());
    }

    if let Some(inner) = rest
        .strip_prefix("simplify(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return TimelineCommandInput::Simplify {
            expr: inner.trim().to_string(),
            aggressive: true,
        };
    }

    if let Some(simplify_rest) = rest.strip_prefix("simplify ") {
        return TimelineCommandInput::Simplify {
            expr: simplify_rest.trim().to_string(),
            aggressive: true,
        };
    }

    TimelineCommandInput::Simplify {
        expr: rest.trim().to_string(),
        aggressive: false,
    }
}

fn evaluate_timeline_simplify_aggressive_input(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
) -> Result<TimelineSimplifyEvalOutput, TimelineEvalError> {
    let mut temp_simplifier = cas_solver::Simplifier::with_default_rules();
    temp_simplifier.set_collect_steps(true);

    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut temp_simplifier.context)
            .map_err(|e| TimelineEvalError::Parse(e.to_string()))?;
        let (simplified_expr, steps) = temp_simplifier.simplify(parsed_expr);
        Ok(TimelineSimplifyEvalOutput {
            parsed_expr,
            simplified_expr,
            steps: cas_solver::to_display_steps(steps),
        })
    })();
    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    result
}

fn evaluate_timeline_simplify_input(
    engine: &mut cas_solver::Engine,
    session: &mut cas_session::SessionState,
    input: &str,
) -> Result<TimelineSimplifyEvalOutput, TimelineEvalError> {
    let was_collecting = engine.simplifier.collect_steps();
    engine.simplifier.set_collect_steps(true);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut engine.simplifier.context)
            .map_err(|e| TimelineEvalError::Parse(e.to_string()))?;
        let req = cas_solver::EvalRequest {
            raw_input: input.to_string(),
            parsed: parsed_expr,
            action: cas_solver::EvalAction::Simplify,
            auto_store: false,
        };
        let output = engine
            .eval(session, req)
            .map_err(|e| TimelineEvalError::Eval(e.to_string()))?;
        let simplified_expr = match output.result {
            cas_solver::EvalResult::Expr(e) => e,
            _ => parsed_expr,
        };
        Ok(TimelineSimplifyEvalOutput {
            parsed_expr,
            simplified_expr,
            steps: output.steps,
        })
    })();
    engine.simplifier.set_collect_steps(was_collecting);
    result
}

fn evaluate_timeline_command_input(
    engine: &mut cas_solver::Engine,
    session: &mut cas_session::SessionState,
    input: &str,
    eval_options: &cas_solver::EvalOptions,
) -> Result<cas_didactic::TimelineCommandOutput, TimelineCommandEvalError> {
    match parse_timeline_command_input(input) {
        TimelineCommandInput::Solve(solve_rest) => {
            cas_solver::evaluate_timeline_solve_with_eval_options(
                &mut engine.simplifier,
                &solve_rest,
                eval_options,
            )
            .map(cas_didactic::TimelineCommandOutput::Solve)
            .map_err(TimelineCommandEvalError::Solve)
        }
        TimelineCommandInput::Simplify { expr, aggressive } => {
            let out = if aggressive {
                evaluate_timeline_simplify_aggressive_input(&mut engine.simplifier, &expr)
            } else {
                evaluate_timeline_simplify_input(engine, session, &expr)
            }
            .map_err(TimelineCommandEvalError::Simplify)?;

            Ok(cas_didactic::TimelineCommandOutput::Simplify(
                cas_didactic::TimelineSimplifyCommandOutput {
                    expr_input: expr,
                    use_aggressive: aggressive,
                    parsed_expr: out.parsed_expr,
                    simplified_expr: out.simplified_expr,
                    steps: out.steps,
                },
            ))
        }
    }
}

fn format_timeline_command_error_message(error: &TimelineCommandEvalError) -> String {
    match error {
        TimelineCommandEvalError::Solve(e) => format_timeline_solve_error_message(e),
        TimelineCommandEvalError::Simplify(e) => format_timeline_eval_error_message(e),
    }
}

fn format_timeline_solve_error_message(error: &cas_solver::TimelineSolveEvalError) -> String {
    match error {
        cas_solver::TimelineSolveEvalError::Prepare(
            cas_solver::SolvePrepareError::ExpectedEquation,
        ) => "Error: Expected an equation for solve timeline, got an expression.\n\
                     Usage: timeline solve <equation>, <variable>\n\
                     Example: timeline solve x + 2 = 5, x"
            .to_string(),
        cas_solver::TimelineSolveEvalError::Prepare(cas_solver::SolvePrepareError::ParseError(
            e,
        )) => {
            format!("Error parsing equation: {}", e)
        }
        cas_solver::TimelineSolveEvalError::Prepare(cas_solver::SolvePrepareError::NoVariable) => {
            "Error: timeline solve found no variable.\n\
                 Use timeline solve <equation>, <variable>"
                .to_string()
        }
        cas_solver::TimelineSolveEvalError::Prepare(
            cas_solver::SolvePrepareError::AmbiguousVariables(vars),
        ) => format!(
            "Error: timeline solve found ambiguous variables {{{}}}.\n\
                 Use timeline solve <equation>, {}",
            vars.join(", "),
            vars.first().unwrap_or(&"x".to_string())
        ),
        cas_solver::TimelineSolveEvalError::Solve(e) => format!("Error solving: {}", e),
    }
}

fn format_timeline_eval_error_message(error: &TimelineEvalError) -> String {
    match error {
        TimelineEvalError::Parse(e) => format!("Parse error: {}", e),
        TimelineEvalError::Eval(e) => format!("Simplification error: {}", e),
    }
}

fn format_equivalence_result_lines(result: &cas_solver::EquivalenceResult) -> Vec<String> {
    match result {
        cas_solver::EquivalenceResult::True => vec!["True".to_string()],
        cas_solver::EquivalenceResult::ConditionalTrue { requires } => {
            let mut lines = vec!["True (conditional)".to_string()];
            lines.extend(format_text_requires_lines(requires));
            lines
        }
        cas_solver::EquivalenceResult::False => vec!["False".to_string()],
        cas_solver::EquivalenceResult::Unknown => {
            vec!["Unknown (cannot prove equivalence)".to_string()]
        }
    }
}

fn format_text_requires_lines(requires: &[String]) -> Vec<String> {
    if requires.is_empty() {
        return Vec::new();
    }

    let mut lines = vec!["ℹ️ Requires:".to_string()];
    for req in requires {
        lines.push(format!("  • {req}"));
    }
    lines
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubstituteRenderMode {
    None,
    Succinct,
    Normal,
    Verbose,
}

fn substitute_render_mode_from_display_mode(mode: SetDisplayMode) -> SubstituteRenderMode {
    match mode {
        SetDisplayMode::None => SubstituteRenderMode::None,
        SetDisplayMode::Succinct => SubstituteRenderMode::Succinct,
        SetDisplayMode::Normal => SubstituteRenderMode::Normal,
        SetDisplayMode::Verbose => SubstituteRenderMode::Verbose,
    }
}

fn should_render_substitute_step(step: &cas_solver::Step, mode: SubstituteRenderMode) -> bool {
    match mode {
        SubstituteRenderMode::None => false,
        SubstituteRenderMode::Verbose => true,
        SubstituteRenderMode::Succinct | SubstituteRenderMode::Normal => {
            if step.get_importance() < cas_solver::ImportanceLevel::Medium {
                return false;
            }
            if let (Some(before), Some(after)) = (step.global_before, step.global_after) {
                if before == after {
                    return false;
                }
            }
            true
        }
    }
}

fn format_substitute_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    output: &SubstituteEvalOutput,
    mode: SubstituteRenderMode,
) -> Vec<String> {
    let display_parts = split_by_comma_ignoring_parens(input);
    let expr_str = display_parts.first().map(|s| s.trim()).unwrap_or_default();
    let target_str = display_parts.get(1).map(|s| s.trim()).unwrap_or_default();
    let replacement_str = display_parts.get(2).map(|s| s.trim()).unwrap_or_default();

    let mut lines = Vec::new();
    if mode != SubstituteRenderMode::None {
        let label = match output.strategy {
            cas_solver::SubstituteStrategy::Variable => "Variable substitution",
            cas_solver::SubstituteStrategy::PowerAware => "Expression substitution",
        };
        lines.push(format!(
            "{label}: {} → {} in {}",
            target_str, replacement_str, expr_str
        ));
    }

    if mode != SubstituteRenderMode::None && !output.steps.is_empty() {
        if mode != SubstituteRenderMode::Succinct {
            lines.push("Steps:".to_string());
        }
        for step in &output.steps {
            if should_render_substitute_step(step, mode) {
                if mode == SubstituteRenderMode::Succinct {
                    lines.push(format!(
                        "-> {}",
                        cas_formatter::DisplayExpr {
                            context,
                            id: step.global_after.unwrap_or(step.after)
                        }
                    ));
                } else {
                    lines.push(format!("  {}  [{}]", step.description, step.rule_name));
                }
            }
        }
    }

    lines.push(format!(
        "Result: {}",
        cas_formatter::DisplayExpr {
            context,
            id: output.simplified_expr
        }
    ));
    lines
}

fn format_explain_gcd_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    steps: &[String],
    value: Option<cas_ast::ExprId>,
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!("Parsed: {}", input));
    lines.push(String::new());
    lines.push("Educational Steps:".to_string());
    lines.push("─".repeat(60));

    for step in steps {
        lines.push(step.clone());
    }

    lines.push("─".repeat(60));
    lines.push(String::new());

    if let Some(result_expr) = value {
        lines.push(format!(
            "Result: {}",
            cas_formatter::DisplayExpr {
                context,
                id: result_expr
            }
        ));
    } else {
        lines.push("Could not compute GCD".to_string());
    }
    lines
}

impl Repl {
    pub(crate) fn timeline_cli_render_to_reply(
        &self,
        render: cas_didactic::TimelineCliRender,
    ) -> ReplReply {
        use std::path::PathBuf;

        match render {
            cas_didactic::TimelineCliRender::NoSteps { lines } => reply_output(lines.join("\n")),
            cas_didactic::TimelineCliRender::Html {
                file_name,
                html,
                lines,
            } => {
                let mut reply = ReplReply::new();
                reply.push(ReplMsg::WriteFile {
                    path: PathBuf::from(file_name),
                    contents: html,
                });
                reply.push(ReplMsg::OpenFile {
                    path: PathBuf::from(file_name),
                });
                for line in lines {
                    reply.push(ReplMsg::output(line));
                }
                reply
            }
        }
    }

    pub(crate) fn handle_equiv_core(&mut self, line: &str) -> ReplReply {
        let rest = extract_equiv_command_tail(line);
        match evaluate_equiv_input(&mut self.core.engine.simplifier, rest) {
            Ok(result) => {
                let lines = format_equivalence_result_lines(&result);
                reply_output(lines.join("\n"))
            }
            Err(error) => reply_output(super::error_render::format_expr_pair_parse_error_message(
                &error, "equiv",
            )),
        }
    }

    pub(crate) fn handle_subst_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        // Format: subst <expr>, <target>, <replacement>
        // Examples:
        //   subst x^4 + x^2 + 1, x^2, y   → y² + y + 1 (power-aware)
        //   subst x^2 + x, x, 3          → 12 (variable substitution)
        let rest = extract_substitute_command_tail(line);
        let render_mode = substitute_render_mode_from_display_mode(
            Self::set_display_mode_from_verbosity(verbosity),
        );
        let output = match evaluate_substitute_and_simplify_input(
            &mut self.core.engine.simplifier,
            rest,
            cas_solver::SubstituteOptions::default(),
        ) {
            Ok(output) => output,
            Err(error) => {
                return reply_output(super::error_render::format_substitute_parse_error_message(
                    &error,
                ))
            }
        };

        let mut lines = format_substitute_eval_lines(
            &self.core.engine.simplifier.context,
            rest,
            &output,
            render_mode,
        );
        cas_solver::clean_result_output_line(&mut lines);
        reply_output(lines.join("\n"))
    }

    pub(crate) fn handle_timeline_core(&mut self, line: &str) -> ReplReply {
        let eval_options = self.core.state.options().clone();
        let rest = extract_timeline_command_tail(line);
        let eval_output = match evaluate_timeline_command_input(
            &mut self.core.engine,
            &mut self.core.state,
            rest,
            &eval_options,
        ) {
            Ok(out) => out,
            Err(error) => return reply_output(format_timeline_command_error_message(&error)),
        };

        let render = cas_didactic::render_timeline_command_cli_output(
            &mut self.core.engine.simplifier.context,
            &eval_output,
            cas_didactic::VerbosityLevel::Normal,
        );

        self.timeline_cli_render_to_reply(render)
    }

    pub(crate) fn handle_visualize_core(&mut self, line: &str) -> ReplReply {
        use std::path::PathBuf;

        let rest = extract_visualize_command_tail(line);
        let parsed_expr = match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => expr,
            Err(e) => return reply_output(format!("Parse error: {e}")),
        };
        let mut viz =
            cas_solver::visualizer::AstVisualizer::new(&self.core.engine.simplifier.context);
        let dot = viz.to_dot(parsed_expr);
        let mut reply = vec![ReplMsg::WriteFile {
            path: PathBuf::from("ast.dot"),
            contents: dot,
        }];
        for line in visualize_output_hint_lines() {
            reply.push(ReplMsg::output(line));
        }
        reply
    }

    pub(crate) fn handle_explain_core(&mut self, line: &str) -> ReplReply {
        let rest = extract_explain_command_tail(line);
        let parsed_expr = match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => expr,
            Err(e) => return reply_output(format!("Parse error: {e}")),
        };
        let expr_data = self.core.engine.simplifier.context.get(parsed_expr).clone();
        let cas_ast::Expr::Function(name_id, args) = expr_data else {
            return reply_output(
                "Explain mode currently only supports function calls\n\
                 Try: explain gcd(48, 18)",
            );
        };
        let name = self
            .core
            .engine
            .simplifier
            .context
            .sym_name(name_id)
            .to_string();
        if name != "gcd" {
            return reply_output(format!(
                "Explain mode not yet implemented for function '{}'\n\
                 Currently supported: gcd",
                name
            ));
        }
        if args.len() != 2 {
            return reply_output("Usage: explain gcd(a, b)");
        }
        let result = cas_solver::number_theory::explain_gcd(
            &mut self.core.engine.simplifier.context,
            args[0],
            args[1],
        );
        let mut lines = format_explain_gcd_eval_lines(
            &self.core.engine.simplifier.context,
            rest,
            &result.steps,
            result.value,
        );
        cas_solver::clean_result_output_line(&mut lines);
        reply_output(lines.join("\n"))
    }
}
