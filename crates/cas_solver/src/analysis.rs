use crate::rationalize::{rationalize_denominator, RationalizeConfig, RationalizeResult};
use crate::{
    apply_weierstrass_recursive, canonical_forms, expand_log_recursive, parse_expr_pair,
    parse_substitute_args, substitute_auto_with_strategy, EquivalenceResult, ParseExprPairError,
    ParseSubstituteArgsError, Simplifier, Step, SubstituteOptions, SubstituteStrategy,
};
use cas_ast::{Expr, ExprId};

/// Error while evaluating unary function commands (e.g. det/trace/transpose).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryFunctionEvalError {
    Parse(String),
}

/// Output of evaluating a unary function call over a parsed expression.
#[derive(Debug, Clone)]
pub struct UnaryFunctionEvalOutput {
    pub parsed_expr: ExprId,
    pub result_expr: ExprId,
    pub steps: Vec<Step>,
}

/// Error while evaluating rationalize commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RationalizeEvalError {
    Parse(String),
}

/// Error while evaluating symbolic transform commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformEvalError {
    Parse(String),
}

/// Error while evaluating timeline simplification commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineEvalError {
    Parse(String),
    Eval(String),
}

/// Error while evaluating full simplify commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullSimplifyEvalError {
    Parse(String),
    Resolve(String),
}

/// Rationalize command result classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RationalizeEvalOutcome {
    Success { simplified_expr: ExprId },
    NotApplicable,
    BudgetExceeded,
}

/// Output of rationalize command evaluation.
#[derive(Debug, Clone)]
pub struct RationalizeEvalOutput {
    pub parsed_expr: ExprId,
    pub normalized_expr: ExprId,
    pub outcome: RationalizeEvalOutcome,
}

/// Output of Weierstrass command evaluation.
#[derive(Debug, Clone)]
pub struct WeierstrassEvalOutput {
    pub parsed_expr: ExprId,
    pub substituted_expr: ExprId,
    pub simplified_expr: ExprId,
}

/// Output of expand_log command evaluation.
#[derive(Debug, Clone)]
pub struct ExpandLogEvalOutput {
    pub parsed_expr: ExprId,
    pub expanded_expr: ExprId,
}

/// Output of telescope command evaluation.
#[derive(Debug, Clone)]
pub struct TelescopeEvalOutput {
    pub parsed_expr: ExprId,
    pub formatted_result: String,
}

/// Error while evaluating explain commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExplainEvalError {
    Parse(String),
    NotFunctionCall,
    UnsupportedFunction(String),
    InvalidArity {
        function: String,
        expected: usize,
        actual: usize,
    },
}

/// Output for `explain gcd(a, b)`.
#[derive(Debug, Clone)]
pub struct ExplainGcdEvalOutput {
    pub parsed_expr: ExprId,
    pub steps: Vec<String>,
    pub value: Option<ExprId>,
}

/// Output for AST visualization command.
#[derive(Debug, Clone)]
pub struct VisualizeEvalOutput {
    pub parsed_expr: ExprId,
    pub dot: String,
}

/// Output of substitution command after applying substitution + simplification.
#[derive(Debug, Clone)]
pub struct SubstituteEvalOutput {
    pub substituted_expr: ExprId,
    pub simplified_expr: ExprId,
    pub strategy: SubstituteStrategy,
    pub steps: Vec<Step>,
}

/// Output of timeline simplification evaluation.
#[derive(Debug, Clone)]
pub struct TimelineSimplifyEvalOutput {
    pub parsed_expr: ExprId,
    pub simplified_expr: ExprId,
    pub steps: crate::DisplayEvalSteps,
}

/// Output of full simplify command evaluation.
#[derive(Debug, Clone)]
pub struct FullSimplifyEvalOutput {
    pub parsed_expr: ExprId,
    pub resolved_expr: ExprId,
    pub simplified_expr: ExprId,
    pub steps: Vec<Step>,
    pub stats: cas_engine::PipelineStats,
}

/// Evaluate equivalence from a REPL-style input:
/// `"<expr1>, <expr2>"`.
pub fn evaluate_equiv_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<EquivalenceResult, ParseExprPairError> {
    let (lhs, rhs) = parse_expr_pair(&mut simplifier.context, input)?;
    Ok(simplifier.are_equivalent_extended(lhs, rhs))
}

/// Evaluate substitution from REPL-style input:
/// `"<expr>, <target>, <replacement>"`.
///
/// Returns the substituted expression id and strategy; caller can decide
/// whether to further simplify and how to render steps.
pub fn evaluate_substitute_input(
    simplifier: &mut Simplifier,
    input: &str,
    options: SubstituteOptions,
) -> Result<(ExprId, SubstituteStrategy), ParseSubstituteArgsError> {
    let (expr, target, replacement) = parse_substitute_args(&mut simplifier.context, input)?;
    Ok(substitute_auto_with_strategy(
        &mut simplifier.context,
        expr,
        target,
        replacement,
        options,
    ))
}

/// Evaluate substitution input and immediately simplify the substituted result.
pub fn evaluate_substitute_and_simplify_input(
    simplifier: &mut Simplifier,
    input: &str,
    options: SubstituteOptions,
) -> Result<SubstituteEvalOutput, ParseSubstituteArgsError> {
    let (substituted_expr, strategy) = evaluate_substitute_input(simplifier, input, options)?;
    let (simplified_expr, steps) = simplifier.simplify(substituted_expr);
    Ok(SubstituteEvalOutput {
        substituted_expr,
        simplified_expr,
        strategy,
        steps,
    })
}

/// Evaluate a unary function command:
/// parse `<input>`, build `function_name(input)`, simplify, and return steps.
pub fn evaluate_unary_function_input(
    simplifier: &mut Simplifier,
    function_name: &str,
    input: &str,
) -> Result<UnaryFunctionEvalOutput, UnaryFunctionEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| UnaryFunctionEvalError::Parse(e.to_string()))?;
    let call_expr = simplifier.context.call(function_name, vec![parsed_expr]);
    let (result_expr, steps) = simplifier.simplify(call_expr);
    Ok(UnaryFunctionEvalOutput {
        parsed_expr,
        result_expr,
        steps,
    })
}

/// Evaluate rationalization command:
/// parse input, normalize canonical form, rationalize denominator, simplify success case.
pub fn evaluate_rationalize_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<RationalizeEvalOutput, RationalizeEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| RationalizeEvalError::Parse(format!("{:?}", e)))?;

    let normalized_expr = canonical_forms::normalize_core(&mut simplifier.context, parsed_expr);
    let config = RationalizeConfig::default();
    let rationalized = rationalize_denominator(&mut simplifier.context, normalized_expr, &config);

    let outcome = match rationalized {
        RationalizeResult::Success(expr) => {
            let (simplified_expr, _) = simplifier.simplify(expr);
            RationalizeEvalOutcome::Success { simplified_expr }
        }
        RationalizeResult::NotApplicable => RationalizeEvalOutcome::NotApplicable,
        RationalizeResult::BudgetExceeded => RationalizeEvalOutcome::BudgetExceeded,
    };

    Ok(RationalizeEvalOutput {
        parsed_expr,
        normalized_expr,
        outcome,
    })
}

/// Evaluate Weierstrass command:
/// parse input, apply recursive substitution, then simplify.
pub fn evaluate_weierstrass_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<WeierstrassEvalOutput, TransformEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| TransformEvalError::Parse(e.to_string()))?;
    let substituted_expr = apply_weierstrass_recursive(&mut simplifier.context, parsed_expr);
    let (simplified_expr, _steps) = simplifier.simplify(substituted_expr);
    Ok(WeierstrassEvalOutput {
        parsed_expr,
        substituted_expr,
        simplified_expr,
    })
}

/// Evaluate expand_log command:
/// parse input and recursively apply log expansion.
pub fn evaluate_expand_log_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<ExpandLogEvalOutput, TransformEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| TransformEvalError::Parse(e.to_string()))?;
    let expanded_expr = expand_log_recursive(&mut simplifier.context, parsed_expr);
    Ok(ExpandLogEvalOutput {
        parsed_expr,
        expanded_expr,
    })
}

/// Evaluate telescope command:
/// parse input and execute telescoping strategy, returning formatted report.
pub fn evaluate_telescope_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<TelescopeEvalOutput, TransformEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| TransformEvalError::Parse(e.to_string()))?;
    let result = crate::telescoping::telescope(&mut simplifier.context, parsed_expr);
    let formatted_result = result.format(&simplifier.context);
    Ok(TelescopeEvalOutput {
        parsed_expr,
        formatted_result,
    })
}

/// Evaluate explain command for `gcd(a, b)`.
pub fn evaluate_explain_gcd_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<ExplainGcdEvalOutput, ExplainEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| ExplainEvalError::Parse(e.to_string()))?;

    let expr_data = simplifier.context.get(parsed_expr).clone();
    let Expr::Function(name_id, args) = expr_data else {
        return Err(ExplainEvalError::NotFunctionCall);
    };
    let name = simplifier.context.sym_name(name_id).to_string();
    if name != "gcd" {
        return Err(ExplainEvalError::UnsupportedFunction(name));
    }
    if args.len() != 2 {
        return Err(ExplainEvalError::InvalidArity {
            function: "gcd".to_string(),
            expected: 2,
            actual: args.len(),
        });
    }

    let result = crate::number_theory::explain_gcd(&mut simplifier.context, args[0], args[1]);
    Ok(ExplainGcdEvalOutput {
        parsed_expr,
        steps: result.steps,
        value: result.value,
    })
}

/// Evaluate visualize command and return DOT graph.
pub fn evaluate_visualize_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<VisualizeEvalOutput, TransformEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| TransformEvalError::Parse(e.to_string()))?;
    let mut viz = crate::visualizer::AstVisualizer::new(&simplifier.context);
    let dot = viz.to_dot(parsed_expr);
    Ok(VisualizeEvalOutput { parsed_expr, dot })
}

/// Evaluate timeline simplify in "aggressive" mode by using a temporary default simplifier.
/// This mirrors CLI `simplify` aggressive flow while preserving caller context.
pub fn evaluate_timeline_simplify_aggressive_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<TimelineSimplifyEvalOutput, TimelineEvalError> {
    let mut temp_simplifier = Simplifier::with_default_rules();
    temp_simplifier.set_collect_steps(true);

    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut temp_simplifier.context)
            .map_err(|e| TimelineEvalError::Parse(e.to_string()))?;
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

/// Evaluate timeline simplify through `Engine::eval` (stateful path).
pub fn evaluate_timeline_simplify_input<S>(
    engine: &mut cas_engine::Engine,
    session: &mut S,
    input: &str,
) -> Result<TimelineSimplifyEvalOutput, TimelineEvalError>
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
    let was_collecting = engine.simplifier.collect_steps();
    engine.simplifier.set_collect_steps(true);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut engine.simplifier.context)
            .map_err(|e| TimelineEvalError::Parse(e.to_string()))?;
        let req = cas_engine::EvalRequest {
            raw_input: input.to_string(),
            parsed: parsed_expr,
            action: cas_engine::EvalAction::Simplify,
            auto_store: false,
        };
        let output = engine
            .eval(session, req)
            .map_err(|e| TimelineEvalError::Eval(e.to_string()))?;
        let simplified_expr = match output.result {
            cas_engine::EvalResult::Expr(e) => e,
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

/// Evaluate full simplify input with aggressive default-rule simplifier.
/// Uses a temporary simplifier with swapped context/profiler and resolves
/// session refs via the provided session.
pub fn evaluate_full_simplify_input<S>(
    engine: &mut cas_engine::Engine,
    session: &S,
    input: &str,
    collect_steps: bool,
) -> Result<FullSimplifyEvalOutput, FullSimplifyEvalError>
where
    S: cas_engine::EvalSession<
        Options = cas_engine::EvalOptions,
        Diagnostics = cas_engine::Diagnostics,
    >,
{
    let mut temp_simplifier = Simplifier::with_default_rules();
    std::mem::swap(&mut engine.simplifier.context, &mut temp_simplifier.context);
    std::mem::swap(
        &mut engine.simplifier.profiler,
        &mut temp_simplifier.profiler,
    );

    let result = (|| {
        let parsed_expr = cas_parser::parse(input, &mut temp_simplifier.context)
            .map_err(|e| FullSimplifyEvalError::Parse(e.to_string()))?;
        let (resolved_expr, _diag, _cache_hits) = session
            .resolve_all_with_diagnostics(&mut temp_simplifier.context, parsed_expr)
            .map_err(|e| FullSimplifyEvalError::Resolve(e.to_string()))?;

        let mut opts = session.options().to_simplify_options();
        opts.collect_steps = collect_steps;
        let (simplified_expr, steps, stats) =
            temp_simplifier.simplify_with_stats(resolved_expr, opts);
        Ok(FullSimplifyEvalOutput {
            parsed_expr,
            resolved_expr,
            simplified_expr,
            steps,
            stats,
        })
    })();

    std::mem::swap(&mut engine.simplifier.context, &mut temp_simplifier.context);
    std::mem::swap(
        &mut engine.simplifier.profiler,
        &mut temp_simplifier.profiler,
    );
    result
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_equiv_input, evaluate_expand_log_input, evaluate_explain_gcd_input,
        evaluate_full_simplify_input, evaluate_rationalize_input,
        evaluate_substitute_and_simplify_input, evaluate_substitute_input,
        evaluate_telescope_input, evaluate_timeline_simplify_input, evaluate_unary_function_input,
        evaluate_visualize_input, evaluate_weierstrass_input, ExplainEvalError,
        FullSimplifyEvalError, RationalizeEvalOutcome, TimelineEvalError, TransformEvalError,
        UnaryFunctionEvalError,
    };
    use crate::SubstituteOptions;
    use crate::{Engine, EquivalenceResult};
    use cas_formatter::DisplayExpr;
    use cas_session::SessionState;

    #[test]
    fn evaluate_equiv_input_true_for_basic_identity() {
        let mut s = crate::Simplifier::with_default_rules();
        let result = evaluate_equiv_input(&mut s, "x + x, 2*x").expect("equiv");
        assert!(matches!(
            result,
            EquivalenceResult::True | EquivalenceResult::ConditionalTrue { .. }
        ));
    }

    #[test]
    fn evaluate_equiv_input_reports_missing_delimiter() {
        let mut s = crate::Simplifier::with_default_rules();
        let err = evaluate_equiv_input(&mut s, "x+x").expect_err("missing delimiter");
        assert!(matches!(err, crate::ParseExprPairError::MissingDelimiter));
    }

    #[test]
    fn evaluate_substitute_input_runs_auto_strategy() {
        let mut s = crate::Simplifier::with_default_rules();
        let (subbed, _strategy) =
            evaluate_substitute_input(&mut s, "x^2 + x, x, 3", SubstituteOptions::default())
                .expect("subst");
        let (simplified, _steps) = s.simplify(subbed);
        let out = format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: simplified
            }
        );
        assert_eq!(out, "12");
    }

    #[test]
    fn evaluate_substitute_and_simplify_input_runs() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_substitute_and_simplify_input(
            &mut s,
            "x^2 + x, x, 3",
            SubstituteOptions::default(),
        )
        .expect("subst");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: out.simplified_expr
            }
        );
        assert_eq!(rendered, "12");
    }

    #[test]
    fn evaluate_unary_function_input_trace_works() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_unary_function_input(&mut s, "trace", "[[1,2],[3,4]]").expect("trace");
        let display = format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: out.result_expr
            }
        );
        assert_eq!(display, "5");
    }

    #[test]
    fn evaluate_unary_function_input_parse_error() {
        let mut s = crate::Simplifier::with_default_rules();
        let err = evaluate_unary_function_input(&mut s, "det", "[[1,2]").expect_err("parse");
        assert!(matches!(err, UnaryFunctionEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_rationalize_input_success() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_rationalize_input(&mut s, "1/(1 + sqrt(2))").expect("rationalize");
        match out.outcome {
            RationalizeEvalOutcome::Success { simplified_expr } => {
                let display = format!(
                    "{}",
                    DisplayExpr {
                        context: &s.context,
                        id: simplified_expr
                    }
                );
                assert_ne!(display, "1/(1 + sqrt(2))");
            }
            _ => panic!("expected success"),
        }
    }

    #[test]
    fn evaluate_weierstrass_input_produces_substitution() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_weierstrass_input(&mut s, "sin(x)").expect("weierstrass");
        let parsed = format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: out.parsed_expr
            }
        );
        let substituted = format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: out.substituted_expr
            }
        );
        assert_eq!(parsed, "sin(x)");
        assert_ne!(substituted, parsed);
    }

    #[test]
    fn evaluate_expand_log_input_expands_simple_log() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_expand_log_input(&mut s, "ln(x*y)").expect("expand_log");
        let display = format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: out.expanded_expr
            }
        );
        assert!(display.contains("ln(x)") && display.contains("ln(y)"));
    }

    #[test]
    fn evaluate_telescope_input_reports_steps() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_telescope_input(&mut s, "1 + 2*cos(x)").expect("telescope");
        assert!(!out.formatted_result.trim().is_empty());
    }

    #[test]
    fn evaluate_transform_inputs_parse_error() {
        let mut s = crate::Simplifier::with_default_rules();
        let err = evaluate_expand_log_input(&mut s, "ln(").expect_err("parse");
        assert!(matches!(err, TransformEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_visualize_input_emits_dot() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_visualize_input(&mut s, "x + 1").expect("viz");
        assert!(out.dot.contains("digraph"));
    }

    #[test]
    fn evaluate_timeline_simplify_input_stateful_runs() {
        let mut engine = Engine::new();
        let mut session = SessionState::new();
        let out =
            evaluate_timeline_simplify_input(&mut engine, &mut session, "x + x").expect("timeline");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: out.simplified_expr
            }
        );
        assert_eq!(rendered, "2 * x");
    }

    #[test]
    fn evaluate_timeline_simplify_input_parse_error() {
        let mut engine = Engine::new();
        let mut session = SessionState::new();
        let err =
            evaluate_timeline_simplify_input(&mut engine, &mut session, "x +").expect_err("parse");
        assert!(matches!(err, TimelineEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_full_simplify_input_runs() {
        let mut engine = Engine::new();
        let session = SessionState::new();
        let out = evaluate_full_simplify_input(&mut engine, &session, "x + x", true).expect("ok");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: out.simplified_expr
            }
        );
        assert_eq!(rendered, "2 * x");
    }

    #[test]
    fn evaluate_full_simplify_input_parse_error() {
        let mut engine = Engine::new();
        let session = SessionState::new();
        let err =
            evaluate_full_simplify_input(&mut engine, &session, "x +", true).expect_err("parse");
        assert!(matches!(err, FullSimplifyEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_explain_gcd_input_runs() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_explain_gcd_input(&mut s, "gcd(48, 18)").expect("explain");
        assert!(!out.steps.is_empty());
    }

    #[test]
    fn evaluate_explain_gcd_input_errors_for_non_function() {
        let mut s = crate::Simplifier::with_default_rules();
        let err = evaluate_explain_gcd_input(&mut s, "x + 1").expect_err("not function");
        assert_eq!(err, ExplainEvalError::NotFunctionCall);
    }
}
