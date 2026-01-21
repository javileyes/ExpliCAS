//! Poly Stats Rule - Inspect metadata of poly_result references.
//!
//! Provides `poly_stats(poly_result(id))` to show polynomial metadata
//! without materializing the full AST.

use crate::poly_store::{thread_local_meta, PolyId};
use crate::rule::{Rewrite, SimpleRule};
use cas_ast::{Context, Expr, ExprId};

/// Rule: poly_stats(poly_result(id)) → metadata display
pub struct PolyStatsRule;

impl SimpleRule for PolyStatsRule {
    fn name(&self) -> &str {
        "poly_stats"
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        // Match: poly_stats(poly_result(id))
        if let Expr::Function(name, args) = ctx.get(expr) {
            if name != "poly_stats" || args.len() != 1 {
                return None;
            }

            let arg = args[0];

            // Extract poly_result(id)
            if let Expr::Function(inner_name, inner_args) = ctx.get(arg) {
                if inner_name != "poly_result" || inner_args.len() != 1 {
                    return None;
                }

                // Extract ID
                if let Expr::Number(n) = ctx.get(inner_args[0]) {
                    if let Ok(id) = n.to_integer().try_into() {
                        let id: PolyId = id;

                        // Get metadata from thread-local store
                        if let Some(meta) = thread_local_meta(id) {
                            // Build result: poly_meta(terms, degree, vars, modulus)
                            let terms = ctx.num(meta.n_terms as i64);
                            let degree = ctx.num(meta.max_total_degree as i64);
                            let nvars = ctx.num(meta.n_vars as i64);
                            let modulus = ctx.num(meta.modulus as i64);

                            let result = ctx.add(Expr::Function(
                                "poly_meta".to_string(),
                                vec![terms, degree, nvars, modulus],
                            ));

                            return Some(Rewrite::new(result).desc(format!(
                                "Poly stats: {} terms, degree {}, {} vars (mod {})",
                                meta.n_terms, meta.max_total_degree, meta.n_vars, meta.modulus
                            )));
                        }
                    }
                }
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
        use crate::poly_modp_conv::VarTable;
        use crate::poly_store::thread_local_get_for_materialize;
        use crate::rules::algebra::gcd_modp::multipoly_modp_to_expr;

        // Match: poly_to_expr(poly_result(id)) or poly_to_expr(poly_result(id), max_terms)
        if let Expr::Function(name, args) = ctx.get(expr).clone() {
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

            // Extract poly_result(id)
            if let Expr::Function(inner_name, inner_args) = ctx.get(arg) {
                if inner_name != "poly_result" || inner_args.len() != 1 {
                    return None;
                }

                // Extract ID
                if let Expr::Number(n) = ctx.get(inner_args[0]) {
                    if let Ok(id) = n.to_integer().try_into() {
                        let id: crate::poly_store::PolyId = id;

                        // Get poly from thread-local store
                        if let Some((meta, poly)) = thread_local_get_for_materialize(id) {
                            // Check limit
                            if meta.n_terms > max_terms {
                                // Return error message
                                let msg = ctx.var(&format!(
                                    "Error: {} terms exceeds limit {}",
                                    meta.n_terms, max_terms
                                ));
                                return Some(Rewrite::new(msg).desc(format!(
                                    "poly_to_expr: {} terms > limit {}",
                                    meta.n_terms, max_terms
                                )));
                            }

                            // Materialize
                            let mut vars = VarTable::new();
                            for name in &meta.var_names {
                                vars.get_or_insert(name);
                            }

                            let materialized = multipoly_modp_to_expr(ctx, &poly, &vars);

                            return Some(
                                Rewrite::new(materialized).desc(format!(
                                    "Materialized polynomial: {} terms",
                                    meta.n_terms
                                )),
                            );
                        }
                    }
                }
            }
        }

        None
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(PolyStatsRule));
    simplifier.add_rule(Box::new(PolyToExprRule));
}
