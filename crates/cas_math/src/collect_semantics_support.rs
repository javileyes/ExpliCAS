//! Semantics-aware wrapper for collect-like-terms.

use cas_ast::{Context, ExprId};

use crate::collect_terms::{CancelledGroup, CombinedGroup};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectSemanticsMode {
    Strict,
    Assume,
    Generic,
}

/// Derive [`CollectSemanticsMode`] from generic mode flags.
pub fn collect_semantics_mode_from_flags(
    assume_mode: bool,
    strict_mode: bool,
) -> CollectSemanticsMode {
    if assume_mode {
        CollectSemanticsMode::Assume
    } else if strict_mode {
        CollectSemanticsMode::Strict
    } else {
        CollectSemanticsMode::Generic
    }
}

/// Whether this mode requires scanning expression for undefined-risk
/// before applying collect semantics.
pub fn collect_semantics_needs_undefined_risk_scan(mode: CollectSemanticsMode) -> bool {
    !matches!(mode, CollectSemanticsMode::Generic)
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
    use super::{
        collect_semantics_mode_from_flags, collect_semantics_needs_undefined_risk_scan,
        collect_with_semantics_mode, CollectSemanticsMode,
    };
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

    #[test]
    fn mode_from_flags_prioritizes_assume_then_strict() {
        assert_eq!(
            collect_semantics_mode_from_flags(true, true),
            CollectSemanticsMode::Assume
        );
        assert_eq!(
            collect_semantics_mode_from_flags(false, true),
            CollectSemanticsMode::Strict
        );
        assert_eq!(
            collect_semantics_mode_from_flags(false, false),
            CollectSemanticsMode::Generic
        );
    }

    #[test]
    fn needs_risk_scan_only_outside_generic() {
        assert!(collect_semantics_needs_undefined_risk_scan(
            CollectSemanticsMode::Strict
        ));
        assert!(collect_semantics_needs_undefined_risk_scan(
            CollectSemanticsMode::Assume
        ));
        assert!(!collect_semantics_needs_undefined_risk_scan(
            CollectSemanticsMode::Generic
        ));
    }
}
