//! Poly Stats Rule - Inspect metadata of poly_result references.
//!
//! Provides `poly_stats(poly_result(id))` to show polynomial metadata
//! without materializing the full AST.

use crate::phase::PhaseMask;
use crate::rule::{Rewrite, SimpleRule};
use cas_ast::{Context, ExprId};
use cas_math::poly_result_calls::{
    rewrite_poly_latex_call_with_default_limit, rewrite_poly_print_call_with_default_limit,
    rewrite_poly_stats_call_with_materialize_limit, rewrite_poly_to_expr_call_with_default_limit,
    DEFAULT_POLY_LATEX_MAX_TERMS, DEFAULT_POLY_PRINT_MAX_TERMS,
    DEFAULT_POLY_STATS_MATERIALIZE_LIMIT, DEFAULT_POLY_TO_EXPR_MAX_TERMS,
};

/// Rule: poly_stats(poly_result(id)) → metadata display
pub struct PolyStatsRule;

impl SimpleRule for PolyStatsRule {
    fn name(&self) -> &str {
        "poly_stats"
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let rewritten = rewrite_poly_stats_call_with_materialize_limit(
            ctx,
            expr,
            DEFAULT_POLY_STATS_MATERIALIZE_LIMIT,
        )?;
        Some(Rewrite::new(rewritten.0).desc(rewritten.1))
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::TRANSFORM
    }
}

/// Rule: poly_to_expr(poly_result(id)) → materialized AST (with limit)
pub struct PolyToExprRule;

impl SimpleRule for PolyToExprRule {
    fn name(&self) -> &str {
        "poly_to_expr"
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let rewritten = rewrite_poly_to_expr_call_with_default_limit(
            ctx,
            expr,
            DEFAULT_POLY_TO_EXPR_MAX_TERMS,
        )?;
        Some(Rewrite::new(rewritten.0).desc(rewritten.1))
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::TRANSFORM
    }
}

/// Rule: poly_print(poly_result(id) [, max_terms]) → formatted string
/// This prints the polynomial directly without AST construction - O(n log n) with sorting
///
/// Usage:
/// - poly_print(poly_result(0))        - print up to 1000 terms
/// - poly_print(poly_result(0), 50)    - print up to 50 terms with truncation message
pub struct PolyPrintRule;

impl SimpleRule for PolyPrintRule {
    fn name(&self) -> &str {
        "poly_print"
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let rewritten =
            rewrite_poly_print_call_with_default_limit(ctx, expr, DEFAULT_POLY_PRINT_MAX_TERMS)?;
        Some(Rewrite::new(rewritten.0).desc(rewritten.1))
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::TRANSFORM
    }
}

/// Rule: poly_latex(poly_result(id) [, max_terms]) → LaTeX formatted string
pub struct PolyLatexRule;

impl SimpleRule for PolyLatexRule {
    fn name(&self) -> &str {
        "poly_latex"
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let rewritten =
            rewrite_poly_latex_call_with_default_limit(ctx, expr, DEFAULT_POLY_LATEX_MAX_TERMS)?;
        Some(Rewrite::new(rewritten.0).desc(rewritten.1))
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::TRANSFORM
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(PolyStatsRule));
    simplifier.add_rule(Box::new(PolyToExprRule));
    simplifier.add_rule(Box::new(PolyPrintRule));
    simplifier.add_rule(Box::new(PolyLatexRule));
}
