//! Substitute API with optional step rendering for upper layers (CLI/JSON).
//!
//! Core substitution logic remains in `cas_math::substitute`.

use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;

pub use cas_math::substitute::SubstituteOptions;

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

#[cfg(test)]
mod tests {
    use super::{
        detect_substitute_strategy, substitute_auto_with_strategy, SubstituteOptions,
        SubstituteStrategy,
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
}
