//! Poly Stats Rule - Inspect metadata of poly_result references.
//!
//! Provides `poly_stats(poly_result(id))` to show polynomial metadata
//! without materializing the full AST.

use crate::rule::{Rewrite, SimpleRule};
use cas_ast::{Context, Expr, ExprId};
use cas_math::poly_store::thread_local_meta;

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
        use cas_math::poly_modp_conv::{multipoly_modp_to_expr, VarTable};
        use cas_math::poly_store::thread_local_get_for_materialize;

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

            // Get poly from thread-local store
            if let Some((meta, poly)) = thread_local_get_for_materialize(id) {
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

                // Materialize
                let mut vars = VarTable::new();
                for name in &meta.var_names {
                    vars.get_or_insert(name);
                }

                let materialized = multipoly_modp_to_expr(ctx, &poly, &vars);

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
        use cas_math::poly_store::thread_local_get_for_materialize;

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

            // Get poly from thread-local store
            if let Some((meta, poly)) = thread_local_get_for_materialize(id) {
                // Format polynomial with truncation
                let formatted = format_poly_with_limit(&poly, &meta.var_names, max_terms);

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

/// Format MultiPolyModP directly to string with optional truncation
/// Terms are sorted by graded lex order (total degree first, then lex)
fn format_poly_with_limit(
    poly: &cas_math::multipoly_modp::MultiPolyModP,
    var_names: &[String],
    max_terms: usize,
) -> String {
    use cas_math::mono::Mono;
    use std::fmt::Write;

    if poly.terms.is_empty() {
        return "0".to_string();
    }

    // Sort terms by graded lex order for consistent output
    let mut sorted_terms: Vec<&(Mono, u64)> = poly.terms.iter().collect();
    sorted_terms.sort_by(|a, b| {
        // First by total degree (descending)
        let deg_cmp = b.0.total_degree().cmp(&a.0.total_degree());
        if deg_cmp != std::cmp::Ordering::Equal {
            return deg_cmp;
        }
        // Then by lex order (descending)
        b.0.cmp(&a.0)
    });

    let total_terms = sorted_terms.len();
    let show_terms = sorted_terms.len().min(max_terms);
    let truncated = total_terms > max_terms;

    let mut result = String::with_capacity(show_terms * 25);

    for (i, (mono, coeff)) in sorted_terms.iter().take(show_terms).enumerate() {
        let is_constant = mono.total_degree() == 0;

        // Handle sign and spacing
        if i == 0 {
            // First term: no leading sign for positive
            if is_constant {
                let _ = write!(result, "{}", coeff);
            } else if *coeff == 1 {
                // coefficient 1 is implicit
            } else {
                let _ = write!(result, "{}", coeff);
            }
        } else {
            // Subsequent terms: always show +
            if is_constant || *coeff != 1 {
                let _ = write!(result, " + {}", coeff);
            } else {
                result.push_str(" + ");
            }
        }

        // Format monomial (variables with exponents)
        let mut first_var = i == 0 && *coeff == 1 && !is_constant;
        for (var_idx, &exp) in mono.0.iter().enumerate() {
            if exp > 0 && var_idx < var_names.len() {
                if !first_var {
                    result.push('·');
                }
                first_var = false;
                result.push_str(&var_names[var_idx]);
                if exp > 1 {
                    let _ = write!(result, "^{}", exp);
                }
            }
        }
    }

    // Add truncation message
    if truncated {
        let remaining = total_terms - max_terms;
        let _ = write!(result, " + ... (+{} more terms)", remaining);
    }

    result
}

/// Rule: poly_latex(poly_result(id) [, max_terms]) → LaTeX formatted string
pub struct PolyLatexRule;

impl SimpleRule for PolyLatexRule {
    fn name(&self) -> &str {
        "poly_latex"
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        use cas_math::poly_store::thread_local_get_for_materialize;

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

            // Get poly from thread-local store
            if let Some((meta, poly)) = thread_local_get_for_materialize(id) {
                let formatted = format_poly_latex(&poly, &meta.var_names, max_terms);
                let result = ctx.var(&formatted);
                return Some(Rewrite::new(result).desc_lazy(|| {
                    format!("LaTeX polynomial: {} terms", meta.n_terms.min(max_terms))
                }));
            }
        }
        None
    }
}

/// Format polynomial as LaTeX
fn format_poly_latex(
    poly: &cas_math::multipoly_modp::MultiPolyModP,
    var_names: &[String],
    max_terms: usize,
) -> String {
    use cas_math::mono::Mono;
    use std::fmt::Write;

    if poly.terms.is_empty() {
        return "0".to_string();
    }

    let mut sorted_terms: Vec<&(Mono, u64)> = poly.terms.iter().collect();
    sorted_terms.sort_by(|a, b| {
        let deg_cmp = b.0.total_degree().cmp(&a.0.total_degree());
        if deg_cmp != std::cmp::Ordering::Equal {
            return deg_cmp;
        }
        b.0.cmp(&a.0)
    });

    let total_terms = sorted_terms.len();
    let show_terms = sorted_terms.len().min(max_terms);
    let truncated = total_terms > max_terms;

    let mut result = String::with_capacity(show_terms * 30);

    for (i, (mono, coeff)) in sorted_terms.iter().take(show_terms).enumerate() {
        let is_constant = mono.total_degree() == 0;

        if i == 0 {
            if is_constant || *coeff != 1 {
                let _ = write!(result, "{}", coeff);
            }
        } else if is_constant || *coeff != 1 {
            let _ = write!(result, " + {}", coeff);
        } else {
            result.push_str(" + ");
        }

        // LaTeX formatting for variables
        for (var_idx, &exp) in mono.0.iter().enumerate() {
            if exp > 0 && var_idx < var_names.len() {
                let var = &var_names[var_idx];
                if exp == 1 {
                    let _ = write!(result, " {}", var);
                } else {
                    let _ = write!(result, " {}^{{{}}}", var, exp);
                }
            }
        }
    }

    if truncated {
        let remaining = total_terms - max_terms;
        let _ = write!(result, " + \\cdots \\text{{(+{} terms)}}", remaining);
    }

    result
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(PolyStatsRule));
    simplifier.add_rule(Box::new(PolyToExprRule));
    simplifier.add_rule(Box::new(PolyPrintRule));
    simplifier.add_rule(Box::new(PolyLatexRule));
}
