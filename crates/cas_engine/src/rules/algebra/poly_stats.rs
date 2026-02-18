//! Poly Stats Rule - Inspect metadata of poly_result references.
//!
//! Provides `poly_stats(poly_result(id))` to show polynomial metadata
//! without materializing the full AST.

use crate::rule::{Rewrite, SimpleRule};
use cas_ast::{Context, Expr, ExprId};
use cas_math::poly_store::{
    materialize_poly_result_expr, render_poly_result, render_poly_result_latex, thread_local_meta,
};

/// Rule: poly_stats(poly_result(id)) → metadata display
pub struct PolyStatsRule;

impl SimpleRule for PolyStatsRule {
    fn name(&self) -> &str {
        "poly_stats"
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        // Match: poly_stats(poly_result(id))
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let name = ctx.sym_name(*fn_id);
            if name != "poly_stats" || args.len() != 1 {
                return None;
            }

            let arg = args[0];

            // Extract poly_result(id) using canonical helper
            let id = cas_math::poly_result::parse_poly_result_id(ctx, arg)?;

            // Get metadata from thread-local store
            if let Some(meta) = thread_local_meta(id) {
                // Build human-friendly result with clear labeling
                // Format: poly_info(id, terms, vars, repr)
                // where repr = "modp" indicates modular arithmetic
                let poly_id = ctx.num(id as i64);
                let terms = ctx.num(meta.n_terms as i64);
                let nvars = ctx.num(meta.n_vars as i64);

                // repr indicator: "modp" for modular, "exact" for exact
                let repr = ctx.var("modp");

                let result = ctx.call("poly_info", vec![poly_id, terms, nvars, repr]);

                // Materialization threshold based on EXPAND_MATERIALIZE_LIMIT
                let can_materialize = meta.n_terms <= 50_000;
                let materialize_note = if can_materialize {
                    format!("materializable via poly_to_expr(poly_result({}))", id)
                } else {
                    "too large to materialize".to_string()
                };

                return Some(Rewrite::new(result).desc_lazy(|| {
                    format!(
                        "Poly #{}: {} terms, {} vars, repr=modp, {}",
                        id, meta.n_terms, meta.n_vars, materialize_note
                    )
                }));
            }
        }

        None
    }
}

/// Rule: poly_to_expr(poly_result(id)) → materialized AST (with limit)
pub struct PolyToExprRule;

impl SimpleRule for PolyToExprRule {
    fn name(&self) -> &str {
        "poly_to_expr"
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        // Match: poly_to_expr(poly_result(id)) or poly_to_expr(poly_result(id), max_terms)
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let (fn_id, args) = (*fn_id, args.clone());
            let name = ctx.sym_name(fn_id);
            if name != "poly_to_expr" || args.is_empty() || args.len() > 2 {
                return None;
            }

            let arg = args[0];
            let max_terms: usize = if args.len() == 2 {
                if let Expr::Number(n) = ctx.get(args[1]) {
                    n.to_integer().try_into().unwrap_or(50_000)
                } else {
                    50_000
                }
            } else {
                50_000 // Default limit
            };

            // Extract poly_result(id) using canonical helper
            let id = cas_math::poly_result::parse_poly_result_id(ctx, arg)?;

            if let Some(meta) = thread_local_meta(id) {
                // Check limit
                if meta.n_terms > max_terms {
                    // Return error message
                    let msg = ctx.var(&format!(
                        "Error: {} terms exceeds limit {}",
                        meta.n_terms, max_terms
                    ));
                    return Some(Rewrite::new(msg).desc_lazy(|| {
                        format!("poly_to_expr: {} terms > limit {}", meta.n_terms, max_terms)
                    }));
                }

                let materialized = materialize_poly_result_expr(ctx, id)?;

                return Some(
                    Rewrite::new(materialized)
                        .desc_lazy(|| format!("Materialized polynomial: {} terms", meta.n_terms)),
                );
            }
        }

        None
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
        // Match: poly_print(poly_result(id)) or poly_print(poly_result(id), max_terms)
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let (fn_id, args) = (*fn_id, args.clone());
            let name = ctx.sym_name(fn_id);
            if name != "poly_print" || args.is_empty() || args.len() > 2 {
                return None;
            }

            let arg = args[0];
            let max_terms: usize = if args.len() == 2 {
                if let Expr::Number(n) = ctx.get(args[1]) {
                    n.to_integer().try_into().unwrap_or(1000)
                } else {
                    1000
                }
            } else {
                1000 // Default limit for printing
            };

            // Extract poly_result(id) using canonical helper
            let id = cas_math::poly_result::parse_poly_result_id(ctx, arg)?;

            if let Some(meta) = thread_local_meta(id) {
                let formatted = render_poly_result(id, max_terms)?;

                // Return as a variable (string representation)
                let result = ctx.var(&formatted);

                let desc = if meta.n_terms > max_terms {
                    format!(
                        "Formatted polynomial: {} of {} terms shown (no AST)",
                        max_terms, meta.n_terms
                    )
                } else {
                    format!("Formatted polynomial: {} terms (no AST)", meta.n_terms)
                };

                return Some(Rewrite::new(result).desc(desc));
            }
        }

        None
    }
}

/// Rule: poly_latex(poly_result(id) [, max_terms]) → LaTeX formatted string
pub struct PolyLatexRule;

impl SimpleRule for PolyLatexRule {
    fn name(&self) -> &str {
        "poly_latex"
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let (fn_id, args) = (*fn_id, args.clone());
            let name = ctx.sym_name(fn_id);
            if name != "poly_latex" || args.is_empty() || args.len() > 2 {
                return None;
            }

            let arg = args[0];
            let max_terms: usize = if args.len() == 2 {
                if let Expr::Number(n) = ctx.get(args[1]) {
                    n.to_integer().try_into().unwrap_or(100)
                } else {
                    100
                }
            } else {
                100 // LaTeX default is smaller
            };

            // Extract poly_result(id) using canonical helper
            let id = cas_math::poly_result::parse_poly_result_id(ctx, arg)?;

            if let Some(meta) = thread_local_meta(id) {
                let formatted = render_poly_result_latex(id, max_terms)?;
                let result = ctx.var(&formatted);
                return Some(Rewrite::new(result).desc_lazy(|| {
                    format!("LaTeX polynomial: {} terms", meta.n_terms.min(max_terms))
                }));
            }
        }
        None
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(PolyStatsRule));
    simplifier.add_rule(Box::new(PolyToExprRule));
    simplifier.add_rule(Box::new(PolyPrintRule));
    simplifier.add_rule(Box::new(PolyLatexRule));
}
