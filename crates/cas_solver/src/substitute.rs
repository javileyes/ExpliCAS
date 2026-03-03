//! Substitute API with optional step rendering for upper layers (CLI/JSON).
//!
//! Core substitution logic remains in `cas_math::substitute`.

use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;

pub use cas_math::substitute::SubstituteOptions;

/// Parse/eval errors for `subst <expr>, <target>, <replacement>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubstituteParseError {
    InvalidArity,
    Expression(String),
    Target(String),
    Replacement(String),
}

/// Evaluated payload for REPL-style `subst` followed by simplify.
#[derive(Debug, Clone)]
pub struct SubstituteSimplifyEvalOutput {
    pub simplified_expr: ExprId,
    pub strategy: SubstituteStrategy,
    pub steps: Vec<crate::Step>,
}

/// A single substitution step for traceability.
#[derive(Clone, Debug)]
pub struct SubstituteStep {
    /// Rule name: "SubstituteExact", "SubstitutePowerMultiple", "SubstitutePowOfTarget"
    pub rule: String,
    /// Expression before substitution (formatted)
    pub before: String,
    /// Expression after substitution (formatted)
    pub after: String,
    /// Optional note (e.g., "n=4, k=2, m=2")
    pub note: Option<String>,
}

/// Result of substitution including optional steps.
#[derive(Clone, Debug)]
pub struct SubstituteResult {
    pub expr: ExprId,
    pub steps: Vec<SubstituteStep>,
}

/// Strategy chosen for substitution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubstituteStrategy {
    /// Direct variable replacement by node id.
    Variable,
    /// Power-aware expression substitution.
    PowerAware,
}

/// Detect which substitution strategy should be used for a parsed target.
pub fn detect_substitute_strategy(ctx: &Context, target: ExprId) -> SubstituteStrategy {
    match ctx.get(target) {
        Expr::Variable(_) => SubstituteStrategy::Variable,
        _ => SubstituteStrategy::PowerAware,
    }
}

/// Perform power-aware substitution.
pub fn substitute_power_aware(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> ExprId {
    cas_math::substitute::substitute_power_aware(ctx, root, target, replacement, opts)
}

/// Perform substitution selecting strategy from target shape.
///
/// - variable target => direct substitute by id
/// - expression target => power-aware substitution
pub fn substitute_auto(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> ExprId {
    match detect_substitute_strategy(ctx, target) {
        SubstituteStrategy::Variable => {
            cas_ast::substitute_expr_by_id(ctx, root, target, replacement)
        }
        SubstituteStrategy::PowerAware => {
            substitute_power_aware(ctx, root, target, replacement, opts)
        }
    }
}

/// Same as [`substitute_auto`], returning the strategy that was applied.
pub fn substitute_auto_with_strategy(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> (ExprId, SubstituteStrategy) {
    let strategy = detect_substitute_strategy(ctx, target);
    let expr = match strategy {
        SubstituteStrategy::Variable => {
            cas_ast::substitute_expr_by_id(ctx, root, target, replacement)
        }
        SubstituteStrategy::PowerAware => {
            substitute_power_aware(ctx, root, target, replacement, opts)
        }
    };
    (expr, strategy)
}

/// Perform power-aware substitution with step collection.
pub fn substitute_with_steps(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> SubstituteResult {
    let trace = cas_math::substitute::substitute_with_trace(ctx, root, target, replacement, opts);
    let steps = trace
        .steps
        .into_iter()
        .map(|step| SubstituteStep {
            rule: step.rule,
            before: format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: step.before,
                }
            ),
            after: format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: step.after,
                }
            ),
            note: step.note,
        })
        .collect();

    SubstituteResult {
        expr: trace.expr,
        steps,
    }
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

/// Parse REPL-like substitute arguments (`expr, target, replacement`) into ids.
pub fn parse_substitute_args(
    ctx: &mut Context,
    input: &str,
) -> Result<(ExprId, ExprId, ExprId), SubstituteParseError> {
    let parts = split_by_comma_ignoring_parens(input);
    if parts.len() != 3 {
        return Err(SubstituteParseError::InvalidArity);
    }

    let expr_str = parts[0].trim();
    let target_str = parts[1].trim();
    let replacement_str = parts[2].trim();

    let expr = cas_parser::parse(expr_str, ctx)
        .map_err(|e| SubstituteParseError::Expression(e.to_string()))?;
    let target = cas_parser::parse(target_str, ctx)
        .map_err(|e| SubstituteParseError::Target(e.to_string()))?;
    let replacement = cas_parser::parse(replacement_str, ctx)
        .map_err(|e| SubstituteParseError::Replacement(e.to_string()))?;

    Ok((expr, target, replacement))
}

/// Parse, substitute, and simplify REPL-style `subst` input.
pub fn evaluate_substitute_and_simplify(
    simplifier: &mut crate::Simplifier,
    input: &str,
    options: SubstituteOptions,
) -> Result<SubstituteSimplifyEvalOutput, SubstituteParseError> {
    let (expr, target, replacement) = parse_substitute_args(&mut simplifier.context, input)?;
    let (substituted_expr, strategy) =
        substitute_auto_with_strategy(&mut simplifier.context, expr, target, replacement, options);
    let (simplified_expr, steps) = simplifier.simplify(substituted_expr);
    Ok(SubstituteSimplifyEvalOutput {
        simplified_expr,
        strategy,
        steps,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        detect_substitute_strategy, evaluate_substitute_and_simplify, parse_substitute_args,
        substitute_auto_with_strategy, SubstituteOptions, SubstituteParseError, SubstituteStrategy,
    };

    #[test]
    fn detect_substitute_strategy_variable_target() {
        let mut ctx = cas_ast::Context::new();
        let target = cas_parser::parse("x", &mut ctx).expect("target parse");
        assert_eq!(
            detect_substitute_strategy(&ctx, target),
            SubstituteStrategy::Variable
        );
    }

    #[test]
    fn detect_substitute_strategy_expression_target() {
        let mut ctx = cas_ast::Context::new();
        let target = cas_parser::parse("x^2", &mut ctx).expect("target parse");
        assert_eq!(
            detect_substitute_strategy(&ctx, target),
            SubstituteStrategy::PowerAware
        );
    }

    #[test]
    fn substitute_auto_with_strategy_uses_variable_path() {
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("x + 1", &mut ctx).expect("expr parse");
        let target = cas_parser::parse("x", &mut ctx).expect("target parse");
        let replacement = cas_parser::parse("3", &mut ctx).expect("replacement parse");

        let (result, strategy) = substitute_auto_with_strategy(
            &mut ctx,
            expr,
            target,
            replacement,
            SubstituteOptions::default(),
        );
        assert_eq!(strategy, SubstituteStrategy::Variable);
        let out = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: result
            }
        );
        assert!(out.contains('3'));
    }

    #[test]
    fn parse_substitute_args_invalid_arity() {
        let mut ctx = cas_ast::Context::new();
        let err = parse_substitute_args(&mut ctx, "x + 1, x").expect_err("invalid arity");
        assert_eq!(err, SubstituteParseError::InvalidArity);
    }

    #[test]
    fn evaluate_substitute_and_simplify_runs() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let out = evaluate_substitute_and_simplify(
            &mut simplifier,
            "x^2 + x, x, 3",
            SubstituteOptions::default(),
        )
        .expect("evaluate");
        let rendered = cas_formatter::render_expr(&simplifier.context, out.simplified_expr);
        assert_eq!(rendered, "12");
    }
}
