use super::*;
use crate::assumption_format;
use crate::result_format;
use cas_ast::{Context, Equation, ExprId};
use cas_formatter::display_transforms::{DisplayTransformRegistry, ScopeTag, ScopedRenderer};

const WEIERSTRASS_USAGE_MESSAGE: &str = "Usage: weierstrass <expression>\n\
                 Description: Apply Weierstrass substitution (t = tan(x/2))\n\
                 Transforms:\n\
                   sin(x) → 2t/(1+t²)\n\
                   cos(x) → (1-t²)/(1+t²)\n\
                   tan(x) → 2t/(1-t²)\n\
                 Example: weierstrass sin(x) + cos(x)";

fn parse_weierstrass_input(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("weierstrass").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest)
    }
}

fn parse_solve_invocation_check(input: &str, default_check_enabled: bool) -> (bool, &str) {
    let trimmed = input.trim();
    if let Some(rest) = trimmed.strip_prefix("--check") {
        (true, rest.trim_start())
    } else {
        (default_check_enabled, trimmed)
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct SolveCommandInput {
    equation: String,
    variable: Option<String>,
}

#[derive(Debug)]
struct SolveCommandEvalOutput {
    var: String,
    original_equation: Option<Equation>,
    output: cas_solver::EvalOutput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SolveCommandEvalError {
    Prepare(cas_solver::SolvePrepareError),
    Eval(String),
}

fn parse_solve_command_input(input: &str) -> SolveCommandInput {
    if let Some((eq, var)) = rsplit_ignoring_parens(input, ',') {
        return SolveCommandInput {
            equation: eq.trim().to_string(),
            variable: Some(var.trim().to_string()),
        };
    }

    if let Some((eq, var)) = rsplit_ignoring_parens(input, ' ') {
        let eq_trim = eq.trim();
        let var_trim = var.trim();

        let has_operators_after_eq = if let Some(eq_pos) = eq_trim.find('=') {
            let after_eq = &eq_trim[eq_pos + 1..];
            after_eq.contains('+')
                || after_eq.contains('-')
                || after_eq.contains('*')
                || after_eq.contains('/')
                || after_eq.contains('^')
        } else {
            false
        };

        if !var_trim.is_empty()
            && var_trim.chars().all(char::is_alphabetic)
            && !eq_trim.ends_with('=')
            && !has_operators_after_eq
        {
            return SolveCommandInput {
                equation: eq_trim.to_string(),
                variable: Some(var_trim.to_string()),
            };
        }
    }

    SolveCommandInput {
        equation: input.to_string(),
        variable: None,
    }
}

fn evaluate_parsed_solve_command_input(
    engine: &mut cas_solver::Engine,
    session: &mut cas_session::SessionState,
    parsed_input: SolveCommandInput,
    auto_store: bool,
) -> Result<SolveCommandEvalOutput, SolveCommandEvalError> {
    let prepared = prepare_solve_eval_request(
        &mut engine.simplifier.context,
        parsed_input.equation.trim(),
        parsed_input.variable,
        auto_store,
    )
    .map_err(SolveCommandEvalError::Prepare)?;

    let output = engine
        .eval(session, prepared.request)
        .map_err(|e| SolveCommandEvalError::Eval(e.to_string()))?;

    Ok(SolveCommandEvalOutput {
        var: prepared.var,
        original_equation: prepared.original_equation,
        output,
    })
}

#[derive(Debug)]
struct PreparedSolveRequest {
    request: cas_solver::EvalRequest,
    var: String,
    original_equation: Option<Equation>,
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

fn resolve_solve_var(
    ctx: &cas_ast::Context,
    parsed_expr: ExprId,
    explicit_var: Option<String>,
) -> Result<String, cas_solver::SolvePrepareError> {
    if let Some(v) = explicit_var {
        if !v.trim().is_empty() {
            return Ok(v);
        }
    }

    match cas_solver::infer_solve_variable(ctx, parsed_expr) {
        Ok(Some(v)) => Ok(v),
        Ok(None) => Err(cas_solver::SolvePrepareError::NoVariable),
        Err(vars) => Err(cas_solver::SolvePrepareError::AmbiguousVariables(vars)),
    }
}

fn prepare_solve_eval_request(
    ctx: &mut cas_ast::Context,
    input: &str,
    explicit_var: Option<String>,
    auto_store: bool,
) -> Result<PreparedSolveRequest, cas_solver::SolvePrepareError> {
    let stmt = parse_statement_or_session_ref(ctx, input)
        .map_err(cas_solver::SolvePrepareError::ParseError)?;

    let original_equation = match &stmt {
        cas_parser::Statement::Equation(eq) => Some(eq.clone()),
        cas_parser::Statement::Expression(_) => None,
    };

    let parsed_expr = match stmt {
        cas_parser::Statement::Equation(eq) => ctx.call("Equal", vec![eq.lhs, eq.rhs]),
        cas_parser::Statement::Expression(e) => e,
    };

    let var = resolve_solve_var(ctx, parsed_expr, explicit_var)?;

    Ok(PreparedSolveRequest {
        request: cas_solver::EvalRequest {
            raw_input: input.to_string(),
            parsed: parsed_expr,
            action: cas_solver::EvalAction::Solve { var: var.clone() },
            auto_store,
        },
        var,
        original_equation,
    })
}

fn format_solve_prepare_error_message(error: &cas_solver::SolvePrepareError) -> String {
    match error {
        cas_solver::SolvePrepareError::ParseError(e) => format!("Parse error: {}", e),
        cas_solver::SolvePrepareError::NoVariable => {
            "Error: solve() found no variable to solve for.\n\
                     Use solve(expr, x) to specify the variable."
                .to_string()
        }
        cas_solver::SolvePrepareError::AmbiguousVariables(vars) => format!(
            "Error: solve() found ambiguous variables {{{}}}.\n\
                     Use solve(expr, {}) or solve(expr, {{{}}}).",
            vars.join(", "),
            vars.first().unwrap_or(&"x".to_string()),
            vars.join(", ")
        ),
        cas_solver::SolvePrepareError::ExpectedEquation => {
            "Parse error: expected equation".to_string()
        }
    }
}

fn format_solve_command_error_message(error: &SolveCommandEvalError) -> String {
    match error {
        SolveCommandEvalError::Prepare(prepare) => format_solve_prepare_error_message(prepare),
        SolveCommandEvalError::Eval(e) => format!("Error: {}", e),
    }
}

fn format_verify_summary_lines(
    ctx: &Context,
    var: &str,
    verify_result: &cas_solver::VerifyResult,
    detail_prefix: &str,
) -> Vec<String> {
    let mut lines = Vec::new();

    match verify_result.summary {
        cas_solver::VerifySummary::AllVerified => {
            lines.push("✓ All solutions verified".to_string());
        }
        cas_solver::VerifySummary::PartiallyVerified => {
            lines.push("⚠ Some solutions verified".to_string());
            for (sol_id, status) in &verify_result.solutions {
                let sol_str = cas_formatter::render_expr(ctx, *sol_id);
                match status {
                    cas_solver::VerifyStatus::Verified => {
                        lines.push(format!("{detail_prefix}✓ {var} = {sol_str} verified"));
                    }
                    cas_solver::VerifyStatus::Unverifiable { reason, .. } => {
                        lines.push(format!("{detail_prefix}⚠ {var} = {sol_str}: {reason}"));
                    }
                    cas_solver::VerifyStatus::NotCheckable { reason } => {
                        lines.push(format!("{detail_prefix}ℹ {var} = {sol_str}: {reason}"));
                    }
                }
            }
        }
        cas_solver::VerifySummary::NoneVerified => {
            lines.push("⚠ No solutions could be verified".to_string());
        }
        cas_solver::VerifySummary::NotCheckable => {
            if let Some(desc) = &verify_result.guard_description {
                lines.push(format!("ℹ {desc}"));
            } else {
                lines.push("ℹ Solution type not checkable".to_string());
            }
        }
        cas_solver::VerifySummary::Empty => {}
    }

    lines
}

fn format_weierstrass_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    substituted_expr: cas_ast::ExprId,
    simplified_expr: cas_ast::ExprId,
) -> Vec<String> {
    vec![
        format!("Parsed: {}", input),
        String::new(),
        "Weierstrass substitution (t = tan(x/2)):".to_string(),
        format!(
            "  {} → {}",
            input,
            cas_formatter::DisplayExpr {
                context,
                id: substituted_expr
            }
        ),
        String::new(),
        "Simplifying...".to_string(),
        format!(
            "Result: {}",
            cas_formatter::DisplayExpr {
                context,
                id: simplified_expr
            }
        ),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SolveCommandRenderConfig {
    show_steps: bool,
    show_verbose_substeps: bool,
    requires_display: cas_solver::RequiresDisplayLevel,
    debug_mode: bool,
    hints_enabled: bool,
    domain_mode: cas_solver::DomainMode,
    check_solutions: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SolveStepVerbosity {
    show_steps: bool,
    show_verbose_substeps: bool,
}

fn solve_step_verbosity_from_display_mode(mode: SetDisplayMode) -> SolveStepVerbosity {
    match mode {
        SetDisplayMode::None => SolveStepVerbosity {
            show_steps: false,
            show_verbose_substeps: false,
        },
        SetDisplayMode::Succinct | SetDisplayMode::Normal => SolveStepVerbosity {
            show_steps: true,
            show_verbose_substeps: false,
        },
        SetDisplayMode::Verbose => SolveStepVerbosity {
            show_steps: true,
            show_verbose_substeps: true,
        },
    }
}

fn solve_render_config_from_eval_options(
    options: &cas_solver::EvalOptions,
    display_mode: SetDisplayMode,
    debug_mode: bool,
) -> SolveCommandRenderConfig {
    let step_verbosity = solve_step_verbosity_from_display_mode(display_mode);
    SolveCommandRenderConfig {
        show_steps: step_verbosity.show_steps,
        show_verbose_substeps: step_verbosity.show_verbose_substeps,
        requires_display: options.requires_display,
        debug_mode,
        hints_enabled: options.hints_enabled,
        domain_mode: options.shared.semantics.domain_mode,
        check_solutions: options.check_solutions,
    }
}

fn format_solve_steps_lines(
    ctx: &Context,
    solve_steps: &[cas_solver::SolveStep],
    output_scopes: &[ScopeTag],
    include_verbose_substeps: bool,
) -> Vec<String> {
    if solve_steps.is_empty() {
        return Vec::new();
    }

    let registry = DisplayTransformRegistry::with_defaults();
    let style = cas_formatter::root_style::StylePreferences::default();
    let renderer = if output_scopes.is_empty() {
        None
    } else {
        Some(ScopedRenderer::new(ctx, output_scopes, &registry, &style))
    };

    let mut lines = Vec::new();
    for (i, step) in solve_steps.iter().enumerate() {
        lines.push(format!("{}. {}", i + 1, step.description));

        let (lhs_str, rhs_str) = if let Some(ref r) = renderer {
            (
                r.render(step.equation_after.lhs),
                r.render(step.equation_after.rhs),
            )
        } else {
            (
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: step.equation_after.lhs,
                }
                .to_string(),
                cas_formatter::DisplayExpr {
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

        if include_verbose_substeps && !step.substeps.is_empty() {
            for (j, substep) in step.substeps.iter().enumerate() {
                let sub_lhs = cas_formatter::DisplayExpr {
                    context: ctx,
                    id: substep.equation_after.lhs,
                }
                .to_string();
                let sub_rhs = cas_formatter::DisplayExpr {
                    context: ctx,
                    id: substep.equation_after.rhs,
                }
                .to_string();
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

    lines
}

fn format_solve_result_line(
    ctx: &Context,
    result: &cas_solver::EvalResult,
    output_scopes: &[ScopeTag],
) -> String {
    match result {
        cas_solver::EvalResult::SolutionSet(solution_set) => {
            format!(
                "Result: {}",
                result_format::display_solution_set(ctx, solution_set)
            )
        }
        cas_solver::EvalResult::Set(solutions) => {
            let sol_strs: Vec<String> = {
                let registry = DisplayTransformRegistry::with_defaults();
                let style = cas_formatter::root_style::StylePreferences::default();
                let renderer = ScopedRenderer::new(ctx, output_scopes, &registry, &style);
                solutions.iter().map(|id| renderer.render(*id)).collect()
            };
            if sol_strs.is_empty() {
                "Result: No solution".to_string()
            } else {
                format!("Result: {{ {} }}", sol_strs.join(", "))
            }
        }
        _ => format!("Result: {:?}", result),
    }
}

fn requires_result_expr_anchor(result: &cas_solver::EvalResult, resolved: ExprId) -> ExprId {
    match result {
        cas_solver::EvalResult::Expr(expr) => *expr,
        cas_solver::EvalResult::Set(values) => *values.first().unwrap_or(&resolved),
        _ => resolved,
    }
}

fn format_solve_command_eval_lines(
    simplifier: &mut cas_solver::Simplifier,
    eval_out: &SolveCommandEvalOutput,
    config: SolveCommandRenderConfig,
) -> Vec<String> {
    let output = &eval_out.output;
    let mut lines: Vec<String> = Vec::new();

    let id_prefix = output
        .stored_id
        .map(|id| format!("#{id}: "))
        .unwrap_or_default();
    lines.push(format!("{}Solving for {}...", id_prefix, eval_out.var));

    lines.extend(assumption_format::format_domain_warning_lines(
        &output.domain_warnings,
        true,
        "⚠ ",
    ));

    let solver_assumption_records =
        cas_solver::assumption_records_from_engine(&output.solver_assumptions);
    if let Some(summary) =
        assumption_format::format_assumption_records_summary(&solver_assumption_records)
    {
        lines.push(format!("⚠ Assumptions: {summary}"));
    }

    if config.show_steps && !output.solve_steps.is_empty() {
        lines.push("Steps:".to_string());
        lines.extend(format_solve_steps_lines(
            &simplifier.context,
            &output.solve_steps,
            &output.output_scopes,
            config.show_verbose_substeps,
        ));
    }

    lines.push(format_solve_result_line(
        &simplifier.context,
        &output.result,
        &output.output_scopes,
    ));

    let result_expr_id = requires_result_expr_anchor(&output.result, output.resolved);
    let requires_lines = assumption_format::format_diagnostics_requires_lines(
        &mut simplifier.context,
        &output.diagnostics,
        Some(result_expr_id),
        config.requires_display,
        config.debug_mode,
    );
    if !requires_lines.is_empty() {
        lines.push("ℹ️ Requires:".to_string());
        lines.extend(requires_lines);
    }

    if config.check_solutions {
        if let cas_solver::EvalResult::SolutionSet(ref solution_set) = output.result {
            if let Some(ref eq) = eval_out.original_equation {
                let verify_result =
                    cas_solver::verify_solution_set(simplifier, eq, &eval_out.var, solution_set);
                lines.extend(format_verify_summary_lines(
                    &simplifier.context,
                    &eval_out.var,
                    &verify_result,
                    "  ",
                ));
            }
        }
    }

    let hints = cas_solver::take_blocked_hints();
    lines.extend(
        assumption_format::format_solve_assumption_and_blocked_sections(
            &simplifier.context,
            &solver_assumption_records,
            &hints,
            assumption_format::SolveAssumptionSectionConfig {
                debug_mode: config.debug_mode,
                hints_enabled: config.hints_enabled,
                domain_mode: config.domain_mode,
            },
        ),
    );

    lines
}

impl Repl {
    /// Handle the 'weierstrass' command for applying Weierstrass substitution
    /// Transforms sin(x), cos(x), tan(x) into rational expressions in t = tan(x/2)
    pub(crate) fn handle_weierstrass_core(&mut self, line: &str) -> ReplReply {
        let Some(rest) = parse_weierstrass_input(line) else {
            return reply_output(WEIERSTRASS_USAGE_MESSAGE);
        };
        let parsed_expr = match cas_parser::parse(rest, &mut self.core.engine.simplifier.context) {
            Ok(expr) => expr,
            Err(e) => return reply_output(format!("Parse error: {e}")),
        };
        let substituted_expr = cas_solver::apply_weierstrass_recursive(
            &mut self.core.engine.simplifier.context,
            parsed_expr,
        );
        let (simplified_expr, _) = self.core.engine.simplifier.simplify(substituted_expr);
        let mut lines = format_weierstrass_eval_lines(
            &self.core.engine.simplifier.context,
            rest,
            substituted_expr,
            simplified_expr,
        );
        result_format::clean_result_output_line(&mut lines);
        reply_output(lines.join("\n"))
    }

    pub(crate) fn handle_solve_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        let options = self.core.state.options().clone();
        let rest = line.strip_prefix("solve").unwrap_or(line).trim();
        let (check_enabled, solve_tail) =
            parse_solve_invocation_check(rest, options.check_solutions);
        let parsed = parse_solve_command_input(solve_tail);

        let eval_output = match evaluate_parsed_solve_command_input(
            &mut self.core.engine,
            &mut self.core.state,
            parsed,
            true,
        ) {
            Ok(output) => output,
            Err(error) => return reply_output(format_solve_command_error_message(&error)),
        };

        let mut render_config = solve_render_config_from_eval_options(
            &options,
            Self::set_display_mode_from_verbosity(verbosity),
            self.core.debug_mode,
        );
        render_config.check_solutions = check_enabled;

        let lines = format_solve_command_eval_lines(
            &mut self.core.engine.simplifier,
            &eval_output,
            render_config,
        );

        reply_output(lines.join("\n"))
    }
}
