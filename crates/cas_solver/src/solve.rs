//! Solver API facade.
//!
//! Keeps `cas_solver` public API stable while centralizing solver entry points
//! in this crate.

use cas_ast::{Equation, ExprId, SolutionSet};
use cas_math::tri_proof::TriProof;
use cas_solver_core::external_proof::map_external_nonzero_status_with;
pub use cas_solver_core::isolation_utils::contains_var;
pub use cas_solver_core::verify_stats;

use crate::input_parse::{parse_statement_or_session_ref, rsplit_ignoring_parens};
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

/// Prepared request for a stateful `solve` eval action.
#[derive(Debug, Clone)]
pub struct PreparedSolveEvalRequest {
    pub request: crate::EvalRequest,
    pub var: String,
    pub original_equation: Option<Equation>,
}

/// Output of evaluating a parsed `solve` command through an engine/session.
#[derive(Debug, Clone)]
pub struct SolveCommandEvalOutput {
    pub var: String,
    pub original_equation: Option<Equation>,
    pub output: crate::EvalOutputView,
}

/// Errors from evaluating a parsed `solve` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveCommandEvalError {
    Prepare(SolvePrepareError),
    Eval(String),
}

/// Parsed REPL `timeline` command:
/// - `timeline solve ...`
/// - `timeline simplify ...` / `timeline simplify(...)`
/// - `timeline <expr>` (default simplify, non-aggressive)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineCommandInput {
    Solve(String),
    Simplify { expr: String, aggressive: bool },
}

/// Result of preparing a REPL `timeline solve` input.
#[derive(Debug)]
struct PreparedTimelineSolve {
    equation: Equation,
    var: String,
}

/// Result of evaluating a REPL `timeline solve` command end-to-end.
#[derive(Debug, Clone)]
pub struct TimelineSolveEvalOutput {
    pub equation: Equation,
    pub var: String,
    pub solution_set: SolutionSet,
    pub display_steps: DisplaySolveSteps,
    pub diagnostics: SolveDiagnostics,
}

/// Result of evaluating a REPL `timeline simplify` command.
#[derive(Debug, Clone)]
pub struct TimelineSimplifyEvalOutput {
    pub parsed_expr: ExprId,
    pub simplified_expr: ExprId,
    pub steps: crate::DisplayEvalSteps,
}

/// Errors while preparing REPL solve/timeline inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolvePrepareError {
    ParseError(String),
    ExpectedEquation,
    NoVariable,
    AmbiguousVariables(Vec<String>),
}

/// Errors while evaluating REPL `timeline solve`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineSolveEvalError {
    Prepare(SolvePrepareError),
    Solve(String),
}

/// Errors while evaluating REPL `timeline simplify`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineSimplifyEvalError {
    Parse(String),
    Eval(String),
}

/// Unified output of evaluating a `timeline` command.
#[derive(Debug, Clone)]
pub enum TimelineCommandEvalOutput {
    Solve(TimelineSolveEvalOutput),
    Simplify {
        expr_input: String,
        aggressive: bool,
        output: TimelineSimplifyEvalOutput,
    },
}

/// Unified error of evaluating a `timeline` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineCommandEvalError {
    Solve(TimelineSolveEvalError),
    Simplify(TimelineSimplifyEvalError),
}

/// Parse REPL `solve`/`timeline solve` argument shape:
/// - `<equation>, <var>`
/// - `<equation> <var>` (when unambiguous)
/// - `<equation>` (variable inferred later)
pub fn parse_solve_command_input(input: &str) -> SolveCommandInput {
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

/// Parse optional `--check` solve flag and return:
/// - whether solution checking should run
/// - the remaining solve argument tail
pub fn parse_solve_invocation_check(input: &str, default_check_enabled: bool) -> (bool, &str) {
    let trimmed = input.trim();
    if let Some(rest) = trimmed.strip_prefix("--check") {
        (true, rest.trim_start())
    } else {
        (default_check_enabled, trimmed)
    }
}

/// Parse REPL `timeline` argument shape.
pub fn parse_timeline_command_input(rest: &str) -> TimelineCommandInput {
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

/// Parse and evaluate `equiv <expr1>, <expr2>` style input.
pub fn evaluate_equiv_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<crate::EquivalenceResult, crate::input_parse::ParseExprPairError> {
    let (lhs, rhs) = crate::input_parse::parse_expr_pair(&mut simplifier.context, input)?;
    Ok(simplifier.are_equivalent_extended(lhs, rhs))
}

/// Evaluate parsed `solve` input against a stateful engine/session pair.
pub fn evaluate_solve_command_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    parsed_input: SolveCommandInput,
    auto_store: bool,
) -> Result<SolveCommandEvalOutput, SolveCommandEvalError>
where
    S: crate::EvalSession<Options = crate::EvalOptions, Diagnostics = crate::Diagnostics>,
    S::Store: crate::EvalStore<
        DomainMode = crate::DomainMode,
        RequiredItem = crate::RequiredItem,
        Step = crate::Step,
        Diagnostics = crate::Diagnostics,
    >,
{
    let PreparedSolveEvalRequest {
        request,
        var,
        original_equation,
    } = prepare_solve_eval_request(
        &mut engine.simplifier.context,
        parsed_input.equation.trim(),
        parsed_input.variable,
        auto_store,
    )
    .map_err(SolveCommandEvalError::Prepare)?;

    let output = engine
        .eval(session, request)
        .map_err(|e| SolveCommandEvalError::Eval(e.to_string()))?;
    let output = crate::eval_output_view(&output);

    Ok(SolveCommandEvalOutput {
        var,
        original_equation,
        output,
    })
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

/// Prepare a REPL `timeline solve` input:
/// - parses an equation (not plain expression)
/// - resolves explicit/inferred solve variable
fn prepare_timeline_solve_input(
    ctx: &mut cas_ast::Context,
    input: &str,
    explicit_var: Option<String>,
) -> Result<PreparedTimelineSolve, SolvePrepareError> {
    let stmt = parse_statement_or_session_ref(ctx, input).map_err(SolvePrepareError::ParseError)?;

    let equation = match stmt {
        cas_parser::Statement::Equation(eq) => eq,
        cas_parser::Statement::Expression(_) => return Err(SolvePrepareError::ExpectedEquation),
    };

    let eq_expr = ctx.add(cas_ast::Expr::Sub(equation.lhs, equation.rhs));
    let var = resolve_solve_var(ctx, eq_expr, explicit_var)?;

    Ok(PreparedTimelineSolve { equation, var })
}

/// Build a stateful `EvalRequest` for `solve` preserving original expression/equation.
///
/// This allows frontends to parse user `solve` input while keeping orchestration
/// logic out of transport/UI layers.
pub fn prepare_solve_eval_request(
    ctx: &mut cas_ast::Context,
    input: &str,
    explicit_var: Option<String>,
    auto_store: bool,
) -> Result<PreparedSolveEvalRequest, SolvePrepareError> {
    let stmt = parse_statement_or_session_ref(ctx, input).map_err(SolvePrepareError::ParseError)?;

    let original_equation = match &stmt {
        cas_parser::Statement::Equation(eq) => Some(eq.clone()),
        cas_parser::Statement::Expression(_) => None,
    };
    let parsed_expr = statement_to_expr_id(ctx, stmt);
    let var = resolve_solve_var(ctx, parsed_expr, explicit_var)?;

    Ok(PreparedSolveEvalRequest {
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

/// Evaluate a REPL `timeline solve` command:
/// parse command input, prepare equation/variable, and solve with display steps.
fn evaluate_timeline_solve_command_input(
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

fn evaluate_timeline_simplify_aggressive(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<TimelineSimplifyEvalOutput, TimelineSimplifyEvalError> {
    let mut temp_simplifier = Simplifier::with_default_rules();
    temp_simplifier.set_collect_steps(true);

    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut temp_simplifier.context)
            .map_err(|e| TimelineSimplifyEvalError::Parse(e.to_string()))?;
        let (simplified_expr, steps) = temp_simplifier.simplify(parsed_expr);
        Ok(TimelineSimplifyEvalOutput {
            parsed_expr,
            simplified_expr,
            steps: crate::to_display_steps(steps),
        })
    })();
    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    result
}

fn evaluate_timeline_simplify_standard<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    input: &str,
) -> Result<TimelineSimplifyEvalOutput, TimelineSimplifyEvalError>
where
    S: crate::EvalSession<Options = crate::EvalOptions, Diagnostics = crate::Diagnostics>,
    S::Store: crate::EvalStore<
        DomainMode = crate::DomainMode,
        RequiredItem = crate::RequiredItem,
        Step = crate::Step,
        Diagnostics = crate::Diagnostics,
    >,
{
    let was_collecting = engine.simplifier.collect_steps();
    engine.simplifier.set_collect_steps(true);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut engine.simplifier.context)
            .map_err(|e| TimelineSimplifyEvalError::Parse(e.to_string()))?;
        let req = crate::EvalRequest {
            raw_input: input.to_string(),
            parsed: parsed_expr,
            action: crate::EvalAction::Simplify,
            auto_store: false,
        };
        let output = engine
            .eval(session, req)
            .map_err(|e| TimelineSimplifyEvalError::Eval(e.to_string()))?;
        let output_view = crate::eval_output_view(&output);
        let simplified_expr = match output_view.result {
            crate::EvalResult::Expr(e) => e,
            _ => parsed_expr,
        };
        Ok(TimelineSimplifyEvalOutput {
            parsed_expr,
            simplified_expr,
            steps: output_view.steps,
        })
    })();
    engine.simplifier.set_collect_steps(was_collecting);
    result
}

/// Evaluate REPL `timeline simplify` command in standard or aggressive mode.
pub fn evaluate_timeline_simplify_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    input: &str,
    aggressive: bool,
) -> Result<TimelineSimplifyEvalOutput, TimelineSimplifyEvalError>
where
    S: crate::EvalSession<Options = crate::EvalOptions, Diagnostics = crate::Diagnostics>,
    S::Store: crate::EvalStore<
        DomainMode = crate::DomainMode,
        RequiredItem = crate::RequiredItem,
        Step = crate::Step,
        Diagnostics = crate::Diagnostics,
    >,
{
    if aggressive {
        evaluate_timeline_simplify_aggressive(&mut engine.simplifier, input)
    } else {
        evaluate_timeline_simplify_standard(engine, session, input)
    }
}

/// Evaluate REPL `timeline` command (solve/simplify) and return typed output.
pub fn evaluate_timeline_command_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    input: &str,
    eval_options: &crate::EvalOptions,
) -> Result<TimelineCommandEvalOutput, TimelineCommandEvalError>
where
    S: crate::EvalSession<Options = crate::EvalOptions, Diagnostics = crate::Diagnostics>,
    S::Store: crate::EvalStore<
        DomainMode = crate::DomainMode,
        RequiredItem = crate::RequiredItem,
        Step = crate::Step,
        Diagnostics = crate::Diagnostics,
    >,
{
    match parse_timeline_command_input(input) {
        TimelineCommandInput::Solve(solve_rest) => evaluate_timeline_solve_with_eval_options(
            &mut engine.simplifier,
            &solve_rest,
            eval_options,
        )
        .map(TimelineCommandEvalOutput::Solve)
        .map_err(TimelineCommandEvalError::Solve),
        TimelineCommandInput::Simplify { expr, aggressive } => {
            evaluate_timeline_simplify_with_session(engine, session, &expr, aggressive)
                .map(|output| TimelineCommandEvalOutput::Simplify {
                    expr_input: expr,
                    aggressive,
                    output,
                })
                .map_err(TimelineCommandEvalError::Simplify)
        }
    }
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
    crate::expand(&mut simplifier.context, expr)
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
        evaluate_equiv_input, evaluate_timeline_command_with_session,
        evaluate_timeline_simplify_with_session, evaluate_timeline_solve_command_input,
        evaluate_timeline_solve_with_eval_options, parse_solve_command_input,
        parse_solve_invocation_check, parse_timeline_command_input, prepare_solve_eval_request,
        prepare_timeline_solve_input, SolveCommandInput, SolvePrepareError, TimelineCommandInput,
        TimelineSolveEvalError,
    };

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
    fn parse_solve_invocation_check_enables_check_flag() {
        let (check, tail) = parse_solve_invocation_check("--check x+1=2, x", false);
        assert!(check);
        assert_eq!(tail, "x+1=2, x");
    }

    #[test]
    fn parse_solve_invocation_check_uses_default_when_flag_absent() {
        let (check, tail) = parse_solve_invocation_check("x+1=2, x", true);
        assert!(check);
        assert_eq!(tail, "x+1=2, x");
    }

    #[test]
    fn parse_timeline_input_solve() {
        let parsed = parse_timeline_command_input("solve x + 2 = 5, x");
        assert_eq!(
            parsed,
            TimelineCommandInput::Solve("x + 2 = 5, x".to_string())
        );
    }

    #[test]
    fn parse_timeline_input_aggressive_simplify() {
        let parsed = parse_timeline_command_input("simplify(x^2 - 1)");
        assert_eq!(
            parsed,
            TimelineCommandInput::Simplify {
                expr: "x^2 - 1".to_string(),
                aggressive: true,
            }
        );
    }

    #[test]
    fn parse_timeline_input_default_simplify() {
        let parsed = parse_timeline_command_input("x + x");
        assert_eq!(
            parsed,
            TimelineCommandInput::Simplify {
                expr: "x + x".to_string(),
                aggressive: false,
            }
        );
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
    fn prepare_solve_eval_request_accepts_expression() {
        let mut ctx = cas_ast::Context::new();
        let prepared = prepare_solve_eval_request(&mut ctx, "x + 2", Some("x".to_string()), false)
            .expect("prepare");

        assert_eq!(prepared.var, "x");
        assert!(prepared.original_equation.is_none());
        match prepared.request.action {
            crate::EvalAction::Solve { var } => assert_eq!(var, "x"),
            _ => panic!("expected solve action"),
        }
    }

    #[test]
    fn prepare_solve_eval_request_keeps_equation_snapshot() {
        let mut ctx = cas_ast::Context::new();
        let prepared =
            prepare_solve_eval_request(&mut ctx, "x + 2 = 5", None, false).expect("prepare");

        assert_eq!(prepared.var, "x");
        assert!(prepared.original_equation.is_some());
        match prepared.request.action {
            crate::EvalAction::Solve { var } => assert_eq!(var, "x"),
            _ => panic!("expected solve action"),
        }
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
    fn evaluate_timeline_simplify_with_session_standard_runs() {
        let mut engine = crate::Engine::new();
        let mut session = cas_session::SessionState::new();
        let out =
            evaluate_timeline_simplify_with_session(&mut engine, &mut session, "x + x", false)
                .expect("timeline simplify");
        assert_eq!(
            cas_formatter::render_expr(&engine.simplifier.context, out.simplified_expr),
            "2 * x"
        );
    }

    #[test]
    fn evaluate_timeline_simplify_with_session_aggressive_runs() {
        let mut engine = crate::Engine::new();
        let mut session = cas_session::SessionState::new();
        let out = evaluate_timeline_simplify_with_session(
            &mut engine,
            &mut session,
            "(x + 1) * (x - 1)",
            true,
        )
        .expect("timeline simplify aggressive");
        let rendered = cas_formatter::render_expr(&engine.simplifier.context, out.simplified_expr);
        assert!(!rendered.is_empty());
    }

    #[test]
    fn evaluate_timeline_command_with_session_simplify_runs() {
        let mut engine = crate::Engine::new();
        let mut session = cas_session::SessionState::new();
        let eval_opts = crate::EvalOptions::default();
        let out = evaluate_timeline_command_with_session(
            &mut engine,
            &mut session,
            "simplify(x + x)",
            &eval_opts,
        )
        .expect("timeline command simplify");

        match out {
            super::TimelineCommandEvalOutput::Simplify {
                expr_input,
                aggressive,
                output,
            } => {
                assert_eq!(expr_input, "x + x");
                assert!(aggressive);
                assert_eq!(
                    cas_formatter::render_expr(&engine.simplifier.context, output.simplified_expr),
                    "2 * x"
                );
            }
            super::TimelineCommandEvalOutput::Solve(_) => panic!("expected simplify output"),
        }
    }

    #[test]
    fn evaluate_equiv_input_true() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let out = evaluate_equiv_input(&mut simplifier, "x + x, 2*x").expect("equiv");
        assert!(matches!(out, crate::EquivalenceResult::True));
    }

    #[test]
    fn evaluate_equiv_input_requires_delimiter() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let err = evaluate_equiv_input(&mut simplifier, "x + x").expect_err("missing delimiter");
        assert_eq!(
            err,
            crate::input_parse::ParseExprPairError::MissingDelimiter
        );
    }
}
