//! Poly Stats Rule - Inspect metadata of poly_result references.
//!
//! Provides `poly_stats(poly_result(id))` to show polynomial metadata
//! without materializing the full AST.

use crate::rule::{Rewrite, SimpleRule};
use cas_ast::{Context, ExprId};
use cas_math::poly_result_calls::{
    build_poly_info_expr, format_materialization_note, try_parse_poly_latex_call,
    try_parse_poly_print_call, try_parse_poly_stats_call, try_parse_poly_to_expr_call,
};
use cas_math::poly_store::{
    materialize_poly_result_expr, render_poly_result, render_poly_result_latex, thread_local_meta,
};

const POLY_STATS_MATERIALIZE_LIMIT: usize = 50_000;
const POLY_TO_EXPR_DEFAULT_MAX_TERMS: usize = 50_000;
const POLY_PRINT_DEFAULT_MAX_TERMS: usize = 1000;
const POLY_LATEX_DEFAULT_MAX_TERMS: usize = 100;

/// Rule: poly_stats(poly_result(id)) → metadata display
pub struct PolyStatsRule;

impl SimpleRule for PolyStatsRule {
    fn name(&self) -> &str {
        "poly_stats"
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let call = try_parse_poly_stats_call(ctx, expr)?;
        let meta = thread_local_meta(call.poly_id)?;
        let result = build_poly_info_expr(ctx, call.poly_id, meta.n_terms, meta.n_vars);
        let materialize_note =
            format_materialization_note(call.poly_id, meta.n_terms, POLY_STATS_MATERIALIZE_LIMIT);
        Some(Rewrite::new(result).desc_lazy(|| {
            format!(
                "Poly #{}: {} terms, {} vars, repr=modp, {}",
                call.poly_id, meta.n_terms, meta.n_vars, materialize_note
            )
        }))
    }
}

/// Rule: poly_to_expr(poly_result(id)) → materialized AST (with limit)
pub struct PolyToExprRule;

impl SimpleRule for PolyToExprRule {
    fn name(&self) -> &str {
        "poly_to_expr"
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let call = try_parse_poly_to_expr_call(ctx, expr, POLY_TO_EXPR_DEFAULT_MAX_TERMS)?;
        let meta = thread_local_meta(call.poly_id)?;
        if meta.n_terms > call.max_terms {
            let msg = ctx.var(&format!(
                "Error: {} terms exceeds limit {}",
                meta.n_terms, call.max_terms
            ));
            return Some(Rewrite::new(msg).desc_lazy(|| {
                format!(
                    "poly_to_expr: {} terms > limit {}",
                    meta.n_terms, call.max_terms
                )
            }));
        }

        let materialized = materialize_poly_result_expr(ctx, call.poly_id)?;
        Some(
            Rewrite::new(materialized)
                .desc_lazy(|| format!("Materialized polynomial: {} terms", meta.n_terms)),
        )
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
        let call = try_parse_poly_print_call(ctx, expr, POLY_PRINT_DEFAULT_MAX_TERMS)?;
        let meta = thread_local_meta(call.poly_id)?;
        let formatted = render_poly_result(call.poly_id, call.max_terms)?;

        let result = ctx.var(&formatted);
        let desc = if meta.n_terms > call.max_terms {
            format!(
                "Formatted polynomial: {} of {} terms shown (no AST)",
                call.max_terms, meta.n_terms
            )
        } else {
            format!("Formatted polynomial: {} terms (no AST)", meta.n_terms)
        };

        Some(Rewrite::new(result).desc(desc))
    }
}

/// Rule: poly_latex(poly_result(id) [, max_terms]) → LaTeX formatted string
pub struct PolyLatexRule;

impl SimpleRule for PolyLatexRule {
    fn name(&self) -> &str {
        "poly_latex"
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let call = try_parse_poly_latex_call(ctx, expr, POLY_LATEX_DEFAULT_MAX_TERMS)?;
        let meta = thread_local_meta(call.poly_id)?;
        let formatted = render_poly_result_latex(call.poly_id, call.max_terms)?;
        let result = ctx.var(&formatted);
        Some(Rewrite::new(result).desc_lazy(|| {
            format!(
                "LaTeX polynomial: {} terms",
                meta.n_terms.min(call.max_terms)
            )
        }))
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(PolyStatsRule));
    simplifier.add_rule(Box::new(PolyToExprRule));
    simplifier.add_rule(Box::new(PolyPrintRule));
    simplifier.add_rule(Box::new(PolyLatexRule));
}
