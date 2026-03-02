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
struct SolveCommandInput {
    pub equation: String,
    pub variable: Option<String>,
}

/// Result of preparing a REPL `timeline solve` input.
#[derive(Debug)]
struct PreparedTimelineSolve {
    equation: Equation,
    var: String,
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

/// Errors while evaluating REPL `timeline solve`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineSolveEvalError {
    Prepare(SolvePrepareError),
    Solve(String),
}

/// Parse REPL `solve`/`timeline solve` argument shape:
/// - `<equation>, <var>`
/// - `<equation> <var>` (when unambiguous)
/// - `<equation>` (variable inferred later)
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
        evaluate_timeline_solve_command_input, evaluate_timeline_solve_with_eval_options,
        parse_solve_command_input, prepare_timeline_solve_input, SolveCommandInput,
        SolvePrepareError, TimelineSolveEvalError,
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
}
