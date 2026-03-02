//! Solver API facade.
//!
//! Keeps `cas_solver` public API stable while centralizing solver entry points
//! in this crate.

use cas_ast::{Equation, ExprId, SolutionSet};
use cas_math::tri_proof::TriProof;
use cas_solver_core::external_proof::map_external_nonzero_status_with;
pub use cas_solver_core::isolation_utils::contains_var;
pub use cas_solver_core::verify_stats;

use crate::Simplifier;

pub use crate::types::{
    DisplaySolveSteps, SolveCtx, SolveDiagnostics, SolveDomainEnv, SolveStep, SolveSubStep,
    SolverOptions,
};

/// Parsed `solve` command input from the REPL:
/// - equation/expression part
/// - optional explicit variable
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveCommandInput {
    pub equation: String,
    pub variable: Option<String>,
}

/// Parsed REPL `solve` invocation, including one-shot flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveInvocationInput {
    pub check_enabled: bool,
    pub solve_input: SolveCommandInput,
}

/// Result of preparing a REPL `solve` request.
#[derive(Debug)]
pub struct PreparedSolveRequest {
    pub request: crate::EvalRequest,
    pub var: String,
    pub original_equation: Option<Equation>,
}

/// Result of preparing a REPL `timeline solve` input.
#[derive(Debug)]
pub struct PreparedTimelineSolve {
    pub equation: Equation,
    pub var: String,
}

/// Result of evaluating a REPL `solve` command end-to-end.
#[derive(Debug)]
pub struct SolveCommandEvalOutput {
    pub var: String,
    pub original_equation: Option<Equation>,
    pub output: crate::EvalOutput,
}

/// Result of evaluating a full REPL `solve` invocation, including one-shot flags.
#[derive(Debug)]
pub struct SolveInvocationEvalOutput {
    pub check_enabled: bool,
    pub eval_output: SolveCommandEvalOutput,
}

/// Result of evaluating a REPL `timeline solve` command end-to-end.
#[derive(Debug)]
pub struct TimelineSolveEvalOutput {
    pub equation: Equation,
    pub var: String,
    pub solution_set: SolutionSet,
    pub display_steps: DisplaySolveSteps,
    pub diagnostics: SolveDiagnostics,
}

/// Errors while preparing REPL solve/timeline inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolvePrepareError {
    ParseError(String),
    ExpectedEquation,
    NoVariable,
    AmbiguousVariables(Vec<String>),
}

/// Errors while evaluating REPL `solve`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveCommandEvalError {
    Prepare(SolvePrepareError),
    Eval(String),
}

/// Errors while evaluating REPL `timeline solve`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineSolveEvalError {
    Prepare(SolvePrepareError),
    Solve(String),
}

/// Render a user-facing message for a `solve` prepare error.
pub fn format_solve_prepare_error_message(error: &SolvePrepareError) -> String {
    match error {
        SolvePrepareError::ParseError(e) => format!("Parse error: {}", e),
        SolvePrepareError::NoVariable => "Error: solve() found no variable to solve for.\n\
                     Use solve(expr, x) to specify the variable."
            .to_string(),
        SolvePrepareError::AmbiguousVariables(vars) => format!(
            "Error: solve() found ambiguous variables {{{}}}.\n\
                     Use solve(expr, {}) or solve(expr, {{{}}}).",
            vars.join(", "),
            vars.first().unwrap_or(&"x".to_string()),
            vars.join(", ")
        ),
        SolvePrepareError::ExpectedEquation => "Parse error: expected equation".to_string(),
    }
}

/// Render a user-facing message for a `solve` command error.
pub fn format_solve_command_error_message(error: &SolveCommandEvalError) -> String {
    match error {
        SolveCommandEvalError::Prepare(prepare) => format_solve_prepare_error_message(prepare),
        SolveCommandEvalError::Eval(e) => format!("Error: {}", e),
    }
}

/// Render a user-facing message for a `timeline solve` command error.
pub fn format_timeline_solve_error_message(error: &TimelineSolveEvalError) -> String {
    match error {
        TimelineSolveEvalError::Prepare(SolvePrepareError::ExpectedEquation) => {
            "Error: Expected an equation for solve timeline, got an expression.\n\
                     Usage: timeline solve <equation>, <variable>\n\
                     Example: timeline solve x + 2 = 5, x"
                .to_string()
        }
        TimelineSolveEvalError::Prepare(SolvePrepareError::ParseError(e)) => {
            format!("Error parsing equation: {}", e)
        }
        TimelineSolveEvalError::Prepare(SolvePrepareError::NoVariable) => {
            "Error: timeline solve found no variable.\n\
                 Use timeline solve <equation>, <variable>"
                .to_string()
        }
        TimelineSolveEvalError::Prepare(SolvePrepareError::AmbiguousVariables(vars)) => format!(
            "Error: timeline solve found ambiguous variables {{{}}}.\n\
                 Use timeline solve <equation>, {}",
            vars.join(", "),
            vars.first().unwrap_or(&"x".to_string())
        ),
        TimelineSolveEvalError::Solve(e) => format!("Error solving: {}", e),
    }
}

/// Parse REPL `solve` invocation flags + arguments.
///
/// Supported shape:
/// - `--check <equation>, <var>`
/// - `<equation>, <var>`
pub fn parse_solve_invocation_input(
    input: &str,
    default_check_enabled: bool,
) -> SolveInvocationInput {
    let trimmed = input.trim();
    let (check_enabled, solve_tail) = if let Some(rest) = trimmed.strip_prefix("--check") {
        (true, rest.trim_start())
    } else {
        (default_check_enabled, trimmed)
    };

    SolveInvocationInput {
        check_enabled,
        solve_input: parse_solve_command_input(solve_tail),
    }
}

/// Parse REPL `solve`/`timeline solve` argument shape:
/// - `<equation>, <var>`
/// - `<equation> <var>` (when unambiguous)
/// - `<equation>` (variable inferred later)
pub fn parse_solve_command_input(input: &str) -> SolveCommandInput {
    if let Some((eq, var)) = crate::rsplit_ignoring_parens(input, ',') {
        return SolveCommandInput {
            equation: eq.trim().to_string(),
            variable: Some(var.trim().to_string()),
        };
    }

    if let Some((eq, var)) = crate::rsplit_ignoring_parens(input, ' ') {
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

/// Infer the variable to solve for when user doesn't specify one explicitly.
///
/// Returns:
/// - `Ok(Some(var))` if exactly one free variable found
/// - `Ok(None)` if no variables found
/// - `Err(vars)` if multiple variables found (ambiguous)
///
/// Filters out known constants (`pi`, `π`, `e`, `i`) and internal symbols
/// (`_*`, `#*`).
pub fn infer_solve_variable(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Result<Option<String>, Vec<String>> {
    let all_vars = cas_ast::collect_variables(ctx, expr);

    let free_vars: Vec<String> = all_vars
        .into_iter()
        .filter(|v| {
            let is_constant = matches!(v.as_str(), "pi" | "π" | "e" | "i");
            let is_internal = v.starts_with('_') || v.starts_with('#');
            !is_constant && !is_internal
        })
        .collect();

    match free_vars.len() {
        0 => Ok(None),
        1 => Ok(free_vars.into_iter().next()),
        _ => {
            let mut sorted = free_vars;
            sorted.sort();
            Err(sorted)
        }
    }
}

/// Prepare a REPL `solve` request:
/// - parses expression/equation (including `#N`)
/// - resolves explicit/inferred solve variable
/// - builds `EvalRequest { action: Solve }`
pub fn prepare_solve_eval_request(
    ctx: &mut cas_ast::Context,
    input: &str,
    explicit_var: Option<String>,
    auto_store: bool,
) -> Result<PreparedSolveRequest, SolvePrepareError> {
    let stmt =
        crate::parse_statement_or_session_ref(ctx, input).map_err(SolvePrepareError::ParseError)?;

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
        request: crate::EvalRequest {
            raw_input: input.to_string(),
            parsed: parsed_expr,
            action: crate::EvalAction::Solve { var: var.clone() },
            auto_store,
        },
        var,
        original_equation,
    })
}

/// Prepare a REPL `timeline solve` input:
/// - parses an equation (not plain expression)
/// - resolves explicit/inferred solve variable
pub fn prepare_timeline_solve_input(
    ctx: &mut cas_ast::Context,
    input: &str,
    explicit_var: Option<String>,
) -> Result<PreparedTimelineSolve, SolvePrepareError> {
    let stmt =
        crate::parse_statement_or_session_ref(ctx, input).map_err(SolvePrepareError::ParseError)?;

    let equation = match stmt {
        cas_parser::Statement::Equation(eq) => eq,
        cas_parser::Statement::Expression(_) => return Err(SolvePrepareError::ExpectedEquation),
    };

    let eq_expr = ctx.add(cas_ast::Expr::Sub(equation.lhs, equation.rhs));
    let var = resolve_solve_var(ctx, eq_expr, explicit_var)?;

    Ok(PreparedTimelineSolve { equation, var })
}

/// Evaluate a REPL `solve` command:
/// parse command input, build eval request, and run `Engine::eval`.
pub fn evaluate_solve_command_input<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    input: &str,
    auto_store: bool,
) -> Result<SolveCommandEvalOutput, SolveCommandEvalError>
where
    S: cas_engine::EvalSession<
        Options = cas_engine::EvalOptions,
        Diagnostics = cas_engine::Diagnostics,
    >,
    S::Store: cas_engine::EvalStore<
        DomainMode = cas_engine::DomainMode,
        RequiredItem = cas_engine::RequiredItem,
        Step = cas_engine::Step,
        Diagnostics = cas_engine::Diagnostics,
    >,
{
    evaluate_parsed_solve_command_input(
        engine,
        session,
        parse_solve_command_input(input),
        auto_store,
    )
}

/// Evaluate a full REPL `solve` invocation:
/// - parse one-shot flags (`--check`)
/// - parse solve command input
/// - evaluate via engine/session
pub fn evaluate_solve_invocation_input<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    input: &str,
    default_check_enabled: bool,
    auto_store: bool,
) -> Result<SolveInvocationEvalOutput, SolveCommandEvalError>
where
    S: cas_engine::EvalSession<
        Options = cas_engine::EvalOptions,
        Diagnostics = cas_engine::Diagnostics,
    >,
    S::Store: cas_engine::EvalStore<
        DomainMode = cas_engine::DomainMode,
        RequiredItem = cas_engine::RequiredItem,
        Step = cas_engine::Step,
        Diagnostics = cas_engine::Diagnostics,
    >,
{
    let parsed = parse_solve_invocation_input(input, default_check_enabled);
    let eval_output =
        evaluate_parsed_solve_command_input(engine, session, parsed.solve_input, auto_store)?;
    Ok(SolveInvocationEvalOutput {
        check_enabled: parsed.check_enabled,
        eval_output,
    })
}

/// Evaluate a parsed REPL `solve` command.
pub fn evaluate_parsed_solve_command_input<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    parsed_input: SolveCommandInput,
    auto_store: bool,
) -> Result<SolveCommandEvalOutput, SolveCommandEvalError>
where
    S: cas_engine::EvalSession<
        Options = cas_engine::EvalOptions,
        Diagnostics = cas_engine::Diagnostics,
    >,
    S::Store: cas_engine::EvalStore<
        DomainMode = cas_engine::DomainMode,
        RequiredItem = cas_engine::RequiredItem,
        Step = cas_engine::Step,
        Diagnostics = cas_engine::Diagnostics,
    >,
{
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

/// Evaluate a REPL `timeline solve` command:
/// parse command input, prepare equation/variable, and solve with display steps.
pub fn evaluate_timeline_solve_command_input(
    simplifier: &mut Simplifier,
    input: &str,
    opts: SolverOptions,
) -> Result<TimelineSolveEvalOutput, TimelineSolveEvalError> {
    let parsed_input = parse_solve_command_input(input);
    let prepared = prepare_timeline_solve_input(
        &mut simplifier.context,
        parsed_input.equation.trim(),
        parsed_input.variable,
    )
    .map_err(TimelineSolveEvalError::Prepare)?;

    let (solution_set, display_steps, diagnostics) =
        solve_with_display_steps(&prepared.equation, &prepared.var, simplifier, opts)
            .map_err(|e| TimelineSolveEvalError::Solve(e.to_string()))?;

    Ok(TimelineSolveEvalOutput {
        equation: prepared.equation,
        var: prepared.var,
        solution_set,
        display_steps,
        diagnostics,
    })
}

/// Evaluate `timeline solve` using engine `EvalOptions` as semantic source.
pub fn evaluate_timeline_solve_with_eval_options(
    simplifier: &mut Simplifier,
    input: &str,
    eval_options: &crate::EvalOptions,
) -> Result<TimelineSolveEvalOutput, TimelineSolveEvalError> {
    simplifier.set_collect_steps(true);
    let opts = crate::SolverOptions::from_eval_options(eval_options);
    evaluate_timeline_solve_command_input(simplifier, input, opts)
}

/// Solve an equation for a variable.
pub fn solve(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), crate::CasError> {
    crate::solve_core::solve(eq, var, simplifier)
}

/// Solve with display-ready steps and diagnostics.
pub fn solve_with_display_steps(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, DisplaySolveSteps, SolveDiagnostics), crate::CasError> {
    crate::solve_core::solve_with_display_steps(eq, var, simplifier, opts)
}

pub(crate) fn simplifier_context(simplifier: &mut Simplifier) -> &cas_ast::Context {
    &simplifier.context
}

pub(crate) fn simplifier_context_mut(simplifier: &mut Simplifier) -> &mut cas_ast::Context {
    &mut simplifier.context
}

pub(crate) fn simplifier_contains_var(
    simplifier: &mut Simplifier,
    expr: ExprId,
    var: &str,
) -> bool {
    contains_var(&simplifier.context, expr, var)
}

pub(crate) fn simplifier_simplify_expr(simplifier: &mut Simplifier, expr: ExprId) -> ExprId {
    simplifier.simplify(expr).0
}

pub(crate) fn simplifier_expand_expr(simplifier: &mut Simplifier, expr: ExprId) -> ExprId {
    crate::expand::expand(&mut simplifier.context, expr)
}

pub(crate) fn simplifier_render_expr(simplifier: &mut Simplifier, expr: ExprId) -> String {
    cas_formatter::render_expr(&simplifier.context, expr)
}

pub(crate) fn context_render_expr(ctx: &cas_ast::Context, expr: ExprId) -> String {
    cas_formatter::render_expr(ctx, expr)
}

pub(crate) fn simplifier_zero_expr(simplifier: &mut Simplifier) -> ExprId {
    simplifier.context.num(0)
}

pub(crate) fn simplifier_collect_steps(simplifier: &mut Simplifier) -> bool {
    simplifier.collect_steps()
}

pub(crate) fn simplifier_prove_nonzero_status(
    simplifier: &mut Simplifier,
    expr: ExprId,
) -> cas_solver_core::linear_solution::NonZeroStatus {
    map_external_nonzero_status_with(
        crate::prove_nonzero(&simplifier.context, expr),
        |proof| matches!(proof, crate::Proof::Proven | crate::Proof::ProvenImplicit),
        |proof| matches!(proof, crate::Proof::Disproven),
    )
}

pub(crate) fn prove_positive_core(
    ctx: &cas_ast::Context,
    expr: ExprId,
    value_domain: crate::ValueDomain,
) -> TriProof {
    match crate::prove_positive(ctx, expr, value_domain) {
        crate::Proof::Proven | crate::Proof::ProvenImplicit => TriProof::Proven,
        crate::Proof::Disproven => TriProof::Disproven,
        crate::Proof::Unknown => TriProof::Unknown,
    }
}

pub(crate) fn simplifier_is_known_negative(simplifier: &mut Simplifier, expr: ExprId) -> bool {
    cas_solver_core::isolation_utils::is_known_negative(&simplifier.context, expr)
}

pub(crate) fn medium_step(description: String, equation_after: Equation) -> SolveStep {
    SolveStep::new(description, equation_after, crate::ImportanceLevel::Medium)
}

fn resolve_solve_var(
    ctx: &cas_ast::Context,
    parsed_expr: ExprId,
    explicit_var: Option<String>,
) -> Result<String, SolvePrepareError> {
    if let Some(v) = explicit_var {
        if !v.trim().is_empty() {
            return Ok(v);
        }
    }

    match infer_solve_variable(ctx, parsed_expr) {
        Ok(Some(v)) => Ok(v),
        Ok(None) => Err(SolvePrepareError::NoVariable),
        Err(vars) => Err(SolvePrepareError::AmbiguousVariables(vars)),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_parsed_solve_command_input, evaluate_solve_command_input,
        evaluate_solve_invocation_input, evaluate_timeline_solve_command_input,
        evaluate_timeline_solve_with_eval_options, format_solve_command_error_message,
        format_solve_prepare_error_message, format_timeline_solve_error_message,
        parse_solve_command_input, parse_solve_invocation_input, prepare_solve_eval_request,
        prepare_timeline_solve_input, SolveCommandEvalError, SolveCommandInput,
        SolveInvocationInput, SolvePrepareError, TimelineSolveEvalError,
    };
    use cas_session::SessionState;

    #[test]
    fn parse_solve_input_with_comma_var() {
        let parsed = parse_solve_command_input("x + 2 = 5, x");
        assert_eq!(
            parsed,
            SolveCommandInput {
                equation: "x + 2 = 5".to_string(),
                variable: Some("x".to_string()),
            }
        );
    }

    #[test]
    fn parse_solve_input_with_space_var() {
        let parsed = parse_solve_command_input("x + 2 = 5 x");
        assert_eq!(
            parsed,
            SolveCommandInput {
                equation: "x + 2 = 5".to_string(),
                variable: Some("x".to_string()),
            }
        );
    }

    #[test]
    fn parse_solve_input_without_var() {
        let parsed = parse_solve_command_input("x + y = 5");
        assert_eq!(
            parsed,
            SolveCommandInput {
                equation: "x + y = 5".to_string(),
                variable: None,
            }
        );
    }

    #[test]
    fn parse_solve_invocation_input_defaults_check_when_flag_missing() {
        let parsed = parse_solve_invocation_input("x + 2 = 5, x", false);
        assert_eq!(
            parsed,
            SolveInvocationInput {
                check_enabled: false,
                solve_input: SolveCommandInput {
                    equation: "x + 2 = 5".to_string(),
                    variable: Some("x".to_string()),
                },
            }
        );
    }

    #[test]
    fn parse_solve_invocation_input_enables_check_with_flag() {
        let parsed = parse_solve_invocation_input("--check x + 2 = 5, x", false);
        assert_eq!(
            parsed,
            SolveInvocationInput {
                check_enabled: true,
                solve_input: SolveCommandInput {
                    equation: "x + 2 = 5".to_string(),
                    variable: Some("x".to_string()),
                },
            }
        );
    }

    #[test]
    fn prepare_solve_eval_request_infers_variable() {
        let mut ctx = cas_ast::Context::new();
        let prepared = prepare_solve_eval_request(&mut ctx, "x+2=5", None, true).expect("prepare");
        assert_eq!(prepared.var, "x");
        assert!(prepared.original_equation.is_some());
        match prepared.request.action {
            crate::EvalAction::Solve { var } => assert_eq!(var, "x"),
            _ => panic!("expected solve action"),
        }
    }

    #[test]
    fn prepare_solve_eval_request_handles_session_ref_expression() {
        let mut ctx = cas_ast::Context::new();
        let prepared = prepare_solve_eval_request(&mut ctx, "#12", Some("x".to_string()), true)
            .expect("prepare");
        assert!(prepared.original_equation.is_none());
        assert_eq!(prepared.var, "x");
    }

    #[test]
    fn prepare_timeline_solve_requires_equation() {
        let mut ctx = cas_ast::Context::new();
        let err = prepare_timeline_solve_input(&mut ctx, "x+1", None).expect_err("expected error");
        assert_eq!(err, SolvePrepareError::ExpectedEquation);
    }

    #[test]
    fn prepare_timeline_solve_infers_variable() {
        let mut ctx = cas_ast::Context::new();
        let prepared = prepare_timeline_solve_input(&mut ctx, "x+2=5", None).expect("prepare");
        assert_eq!(prepared.var, "x");
    }

    #[test]
    fn evaluate_solve_command_input_runs() {
        let mut engine = crate::Engine::new();
        let mut session = SessionState::new();
        let out = evaluate_solve_command_input(&mut engine, &mut session, "x + 2 = 5, x", true)
            .expect("solve");
        assert_eq!(out.var, "x");
        assert!(out.original_equation.is_some());
    }

    #[test]
    fn evaluate_parsed_solve_command_input_runs() {
        let mut engine = crate::Engine::new();
        let mut session = SessionState::new();
        let out = evaluate_parsed_solve_command_input(
            &mut engine,
            &mut session,
            SolveCommandInput {
                equation: "x + 2 = 5".to_string(),
                variable: Some("x".to_string()),
            },
            true,
        )
        .expect("solve");
        assert_eq!(out.var, "x");
        assert!(out.original_equation.is_some());
    }

    #[test]
    fn evaluate_solve_command_input_prepare_error() {
        let mut engine = crate::Engine::new();
        let mut session = SessionState::new();
        let err = evaluate_solve_command_input(&mut engine, &mut session, "x +", true)
            .expect_err("prepare error");
        assert!(matches!(
            err,
            SolveCommandEvalError::Prepare(SolvePrepareError::ParseError(_))
        ));
    }

    #[test]
    fn evaluate_solve_invocation_input_applies_check_flag() {
        let mut engine = crate::Engine::new();
        let mut session = SessionState::new();
        let out = evaluate_solve_invocation_input(
            &mut engine,
            &mut session,
            "--check x + 2 = 5, x",
            false,
            true,
        )
        .expect("solve invocation");
        assert!(out.check_enabled);
    }

    #[test]
    fn evaluate_timeline_solve_command_input_runs() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let out = evaluate_timeline_solve_command_input(
            &mut simplifier,
            "x + 2 = 5, x",
            crate::SolverOptions::default(),
        )
        .expect("timeline solve");
        assert_eq!(out.var, "x");
        assert!(!matches!(out.solution_set, cas_ast::SolutionSet::Empty));
    }

    #[test]
    fn evaluate_timeline_solve_command_input_requires_equation() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let err = evaluate_timeline_solve_command_input(
            &mut simplifier,
            "x + 2",
            crate::SolverOptions::default(),
        )
        .expect_err("expected equation error");
        assert_eq!(
            err,
            TimelineSolveEvalError::Prepare(SolvePrepareError::ExpectedEquation)
        );
    }

    #[test]
    fn evaluate_timeline_solve_with_eval_options_runs() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let eval_opts = crate::EvalOptions::default();
        let out =
            evaluate_timeline_solve_with_eval_options(&mut simplifier, "x + 2 = 5, x", &eval_opts)
                .expect("timeline solve");
        assert_eq!(out.var, "x");
    }

    #[test]
    fn format_solve_prepare_error_message_reports_ambiguous_variables() {
        let msg = format_solve_prepare_error_message(&SolvePrepareError::AmbiguousVariables(vec![
            "x".to_string(),
            "y".to_string(),
        ]));
        assert!(msg.contains("ambiguous variables {x, y}"));
    }

    #[test]
    fn format_solve_command_error_message_wraps_eval_errors() {
        let msg = format_solve_command_error_message(&SolveCommandEvalError::Eval(
            "solver failed".to_string(),
        ));
        assert_eq!(msg, "Error: solver failed");
    }

    #[test]
    fn format_timeline_solve_error_message_reports_expected_equation() {
        let msg = format_timeline_solve_error_message(&TimelineSolveEvalError::Prepare(
            SolvePrepareError::ExpectedEquation,
        ));
        assert!(msg.contains("Expected an equation for solve timeline"));
    }
}
