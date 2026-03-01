use crate::parent_context::ParentContext;
use cas_ast::{Context, ExprId};
use cas_math::collect_rule_support::CollectRulePlan;
use cas_math::collect_semantics_support::{
    collect_semantics_mode_from_flags, collect_semantics_needs_undefined_risk_scan,
    collect_with_semantics_mode, CollectSemanticsMode, CollectSemanticsResult,
};

/// Result of a semantics-aware collection operation.
/// Contains the new expression and tracking of what was cancelled/combined.
pub(crate) type CollectResult = CollectSemanticsResult;

/// Check if an expression contains any Div with a denominator that is not proven non-zero.
/// This indicates "undefined risk" - the expression could be undefined at some points.
pub(crate) fn has_undefined_risk(ctx: &Context, expr: ExprId) -> bool {
    use crate::domain::Proof;
    use crate::helpers::prove_nonzero;

    cas_math::undefined_risk_support::has_undefined_risk_with(ctx, expr, |core_ctx, den| {
        prove_nonzero(core_ctx, den) == Proof::Proven
    })
}

/// Collects like terms with domain_mode awareness.
/// In Strict mode, refuses to cancel terms that might be undefined.
/// In Assume mode, cancels with a warning.
/// In Generic mode, cancels unconditionally.
///
/// Returns None if the result would be identical to input or if blocked by Strict mode.
pub(crate) fn collect_with_semantics(
    ctx: &mut Context,
    expr: ExprId,
    parent_ctx: &ParentContext,
) -> Option<CollectResult> {
    let (mode, risk) = resolve_mode_and_risk(ctx, expr, parent_ctx);

    collect_with_semantics_mode(ctx, expr, mode, risk)
}

/// Build a didactic rewrite plan for CombineLikeTerms rule.
pub(crate) fn plan_collect_rule_rewrite(
    ctx: &mut Context,
    expr: ExprId,
    parent_ctx: &ParentContext,
) -> Option<CollectRulePlan> {
    let (mode, risk) = resolve_mode_and_risk(ctx, expr, parent_ctx);
    cas_math::collect_rule_support::try_plan_collect_rule_expr(ctx, expr, mode, risk)
}

/// Collects like terms in an expression using Generic mode semantics.
/// e.g. 2*x + 3*x -> 5*x
///      x + x -> 2*x
///      x^2 + 2*x^2 -> 3*x^2
pub(crate) fn collect(ctx: &mut Context, expr: ExprId) -> ExprId {
    // Generic mode keeps legacy behavior (no blocking, no warnings).
    match collect_with_semantics_mode(ctx, expr, CollectSemanticsMode::Generic, false) {
        Some(result) => result.new_expr,
        None => expr, // No change or blocked
    }
}

fn resolve_mode_and_risk(
    ctx: &Context,
    expr: ExprId,
    parent_ctx: &ParentContext,
) -> (CollectSemanticsMode, bool) {
    let domain_mode = parent_ctx.domain_mode();
    let mode = collect_semantics_mode_from_flags(
        matches!(domain_mode, crate::DomainMode::Assume),
        matches!(domain_mode, crate::DomainMode::Strict),
    );
    let risk = if collect_semantics_needs_undefined_risk_scan(mode) {
        has_undefined_risk(ctx, expr)
    } else {
        false
    };
    (mode, risk)
}

/// Simplify numeric sums in exponents throughout an expression tree.
/// e.g., x^(1/2 + 1/3) → x^(5/6)
/// This is applied during the collect phase for early, visible simplification.
#[allow(dead_code)] // Infrastructure: tested, reserved for collect pipeline
pub fn simplify_numeric_exponents(ctx: &mut Context, expr: ExprId) -> ExprId {
    cas_math::collect_terms::simplify_numeric_exponents(ctx, expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn s(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn test_collect_integers() {
        let mut ctx = Context::new();
        let expr = parse("1 + 2", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        assert_eq!(s(&ctx, res), "3");
    }

    #[test]
    fn test_collect_variables() {
        let mut ctx = Context::new();
        let expr = parse("x + x", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        assert_eq!(s(&ctx, res), "2 * x");
    }

    #[test]
    fn test_collect_mixed() {
        let mut ctx = Context::new();
        let expr = parse("2*x + 3*y + 4*x", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        // Order depends on implementation, but should have 6*x and 3*y
        let res_str = s(&ctx, res);
        assert!(res_str.contains("6 * x"));
        assert!(res_str.contains("3 * y"));
    }

    #[test]
    fn test_collect_cancel() {
        let mut ctx = Context::new();
        let expr = parse("x - x", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        assert_eq!(s(&ctx, res), "0");
    }

    #[test]
    fn test_collect_powers() {
        let mut ctx = Context::new();
        let expr = parse("x^2 + 2*x^2", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        assert_eq!(s(&ctx, res), "3 * x^2");
    }

    #[test]
    fn test_simplify_numeric_exponents() {
        let mut ctx = Context::new();
        // x^(1/2 + 1/3) should become x^(5/6)
        let expr = parse("x^(1/2 + 1/3)", &mut ctx).unwrap();
        let res = simplify_numeric_exponents(&mut ctx, expr);
        // The result should be different (simplified)
        assert_ne!(res, expr, "Expression should be simplified");
        assert_eq!(s(&ctx, res), "x^(5/6)");
    }

    #[test]
    fn test_collect_double_negation() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).unwrap();
        let neg_x = ctx.add(Expr::Neg(x));
        let neg_neg_x = ctx.add(Expr::Neg(neg_x));
        let res = collect(&mut ctx, neg_neg_x);
        assert_eq!(s(&ctx, res), "x");
    }

    #[test]
    fn test_collect_sub_neg() {
        let mut ctx = Context::new();
        let expr = parse("a - (-b)", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        let res_str = s(&ctx, res);
        assert!(res_str == "a + b" || res_str == "b + a");
    }

    #[test]
    fn test_collect_nested_neg_add() {
        let mut ctx = Context::new();
        let expr = parse("a + -(-b)", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        let res_str = s(&ctx, res);
        assert!(res_str == "a + b" || res_str == "b + a");
    }

    #[test]
    fn test_collect_neg_neg_cos() {
        let mut ctx = Context::new();
        let expr = parse("-(-cos(x))", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        assert_eq!(s(&ctx, res), "cos(x)");
    }

    #[test]
    fn test_collect_sub_neg_cos() {
        let mut ctx = Context::new();
        let expr = parse("-3 - (-cos(x))", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        let res_str = s(&ctx, res);
        assert!(res_str.contains("cos(x)"));
        assert!(!res_str.contains("- -"));
        assert!(!res_str.contains("- (-"));
    }

    #[test]
    fn test_collect_user_repro() {
        let mut ctx = Context::new();
        let expr = parse("8 * sin(x)^4 - (3 - 4 * cos(2 * x) + cos(4 * x))", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        let res_str = s(&ctx, res);
        assert!(!res_str.contains("- -cos"));
        assert!(!res_str.contains("- (-cos"));
        assert!(res_str.contains("cos(4 * x)"));
        assert!(res_str.contains("3"));
    }
}
