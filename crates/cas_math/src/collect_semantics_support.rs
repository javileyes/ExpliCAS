//! Semantics-aware wrapper for collect-like-terms.

use cas_ast::{Context, ExprId};

use crate::collect_terms::{CancelledGroup, CombinedGroup};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectSemanticsMode {
    Strict,
    Assume,
    Generic,
}

#[derive(Debug, Clone)]
pub struct CollectSemanticsResult {
    pub new_expr: ExprId,
    pub assumption: Option<String>,
    pub cancelled: Vec<CancelledGroup>,
    pub combined: Vec<CombinedGroup>,
}

/// Apply collect semantics policy to an expression.
///
/// `undefined_risk` should be computed by caller semantics (e.g. denominator
/// definedness oracle).
pub fn collect_with_semantics_mode(
    ctx: &mut Context,
    expr: ExprId,
    mode: CollectSemanticsMode,
    undefined_risk: bool,
) -> Option<CollectSemanticsResult> {
    // CRITICAL: Do NOT collect non-commutative expressions (e.g., matrices)
    if !ctx.is_mul_commutative(expr) {
        return None;
    }

    let (allowed, assumption) = match mode {
        CollectSemanticsMode::Strict => {
            if undefined_risk {
                (false, None)
            } else {
                (true, None)
            }
        }
        CollectSemanticsMode::Assume => {
            let assumption = if undefined_risk {
                Some("Assuming expression is defined (denominators ≠ 0)".to_string())
            } else {
                None
            };
            (true, assumption)
        }
        CollectSemanticsMode::Generic => (true, None),
    };

    if !allowed {
        return None;
    }

    let impl_result = crate::collect_terms::collect_impl(ctx, expr);
    if impl_result.new_expr == expr {
        return None;
    }

    Some(CollectSemanticsResult {
        new_expr: impl_result.new_expr,
        assumption,
        cancelled: impl_result.cancelled,
        combined: impl_result.combined,
    })
}

#[cfg(test)]
mod tests {
    use super::{collect_with_semantics_mode, CollectSemanticsMode};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn strict_blocks_when_undefined_risk() {
        let mut ctx = Context::new();
        let expr = parse("x/(x+1) - x/(x+1)", &mut ctx).expect("parse");
        let out = collect_with_semantics_mode(&mut ctx, expr, CollectSemanticsMode::Strict, true);
        assert!(out.is_none());
    }

    #[test]
    fn assume_allows_and_adds_assumption() {
        let mut ctx = Context::new();
        let expr = parse("x + x", &mut ctx).expect("parse");
        let out = collect_with_semantics_mode(&mut ctx, expr, CollectSemanticsMode::Assume, true)
            .expect("out");
        assert!(out.assumption.is_some());
    }

    #[test]
    fn generic_allows_without_assumption() {
        let mut ctx = Context::new();
        let expr = parse("x + x", &mut ctx).expect("parse");
        let out = collect_with_semantics_mode(&mut ctx, expr, CollectSemanticsMode::Generic, true)
            .expect("out");
        assert!(out.assumption.is_none());
    }
}
