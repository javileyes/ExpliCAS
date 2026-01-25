//! Formatting utilities for JSON CLI.
//!
//! Provides truncated expression formatting and AST statistics
//! to avoid flooding output with huge expressions.

use cas_ast::{Case, Context, DisplayExpr, Expr, ExprId, SolutionSet};

use crate::json_types::ExprStatsJson;

/// Check if a conditional case is an "otherwise" that only contains Residual.
///
/// These cases don't add useful information to the user (they just say
/// "otherwise, solve this equation" which is redundant) and should be skipped
/// in both REPL and JSON output for cleaner presentation.
///
/// This is the single source of truth for this logic - used by:
/// - `commands/eval_json.rs` (format_solution_set, solution_set_to_latex)
/// - `repl/free_fns.rs` (display_solution_set)
pub fn is_pure_residual_otherwise(case: &Case) -> bool {
    case.when.is_empty() && matches!(&case.then.solutions, SolutionSet::Residual(_))
}

/// Format an expression with a character limit.
///
/// Returns (formatted_string, was_truncated, original_char_count).
/// Automatically renders poly_result expressions as formatted polynomials.
pub fn format_expr_limited(ctx: &Context, expr: ExprId, max_chars: usize) -> (String, bool, usize) {
    // Try to render as poly_result first (fast path for large polynomials)
    if let Some(poly_str) = cas_engine::poly_store::try_render_poly_result(ctx, expr) {
        let len = poly_str.chars().count();
        if len <= max_chars {
            return (poly_str, false, len);
        }
        // Truncate poly string
        let truncated: String = poly_str.chars().take(max_chars).collect();
        return (format!("{truncated} … <truncated>"), true, len);
    }

    // Standard expression formatting
    let full = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: expr
        }
    );
    let len = full.chars().count();

    if len <= max_chars {
        return (full, false, len);
    }

    // Truncate and add indicator
    let truncated: String = full.chars().take(max_chars).collect();
    (format!("{truncated} … <truncated>"), true, len)
}

/// Compute expression statistics (node count, depth, and term count).
pub fn expr_stats(ctx: &Context, expr: ExprId) -> ExprStatsJson {
    let (node_count, depth) = count_nodes_and_depth(ctx, expr, 0);

    // Try to get term count - first check if it's a poly_result
    let term_count =
        cas_engine::poly_store::try_get_poly_result_term_count(ctx, expr).or_else(|| {
            // For large Add chains, count additive terms (top-level + structure)
            count_add_terms(ctx, expr)
        });

    ExprStatsJson {
        node_count,
        depth,
        term_count,
    }
}

/// Count additive terms in an expression (top-level Add/Sub chain).
/// Returns Some(count) only if there are multiple terms, None for simple expressions.
/// Uses an iterative approach to avoid stack overflow with large expressions.
fn count_add_terms(ctx: &Context, expr: ExprId) -> Option<usize> {
    // Unwrap __hold wrapper if present (used to prevent further simplification of large expressions)
    let inner_expr = match ctx.get(expr) {
        Expr::Function(name, args) if ctx.sym_name(*name) == "__hold" && args.len() == 1 => args[0],
        _ => expr,
    };

    // Check if the inner expression is an Add/Sub chain
    let is_add_sub = matches!(ctx.get(inner_expr), Expr::Add(_, _) | Expr::Sub(_, _));
    if !is_add_sub {
        return None;
    }

    let mut count = 0usize;
    let mut stack = vec![inner_expr];

    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Add(l, r) | Expr::Sub(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            _ => {
                count += 1;
            }
        }
    }

    // Only return count if expression has multiple terms
    if count > 1 {
        Some(count)
    } else {
        None
    }
}

/// Compute nodes and max depth.
///
/// Wrapper calling canonical `cas_ast::traversal::count_nodes_and_max_depth`.
/// (See POLICY.md "Traversal Contract")
fn count_nodes_and_depth(ctx: &Context, expr: ExprId, _current_depth: usize) -> (usize, usize) {
    cas_ast::traversal::count_nodes_and_max_depth(ctx, expr)
}

/// Compute a simple hash of an expression for identity comparison.
///
/// Uses a fast, non-cryptographic hash based on expression structure.
pub fn expr_hash(ctx: &Context, expr: ExprId) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    hash_expr_recursive(ctx, expr, &mut hasher);
    format!("{:016x}", hasher.finish())
}

fn hash_expr_recursive<H: std::hash::Hasher>(ctx: &Context, expr: ExprId, hasher: &mut H) {
    use std::hash::Hash;

    // Hash a tag for each variant type
    match ctx.get(expr) {
        Expr::Number(n) => {
            0u8.hash(hasher);
            n.numer().to_string().hash(hasher);
            n.denom().to_string().hash(hasher);
        }
        Expr::Variable(name) => {
            1u8.hash(hasher);
            name.hash(hasher);
        }
        Expr::Constant(c) => {
            2u8.hash(hasher);
            format!("{:?}", c).hash(hasher);
        }
        Expr::SessionRef(id) => {
            11u8.hash(hasher);
            id.hash(hasher);
        }
        Expr::Add(l, r) => {
            3u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Sub(l, r) => {
            4u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Mul(l, r) => {
            5u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Div(l, r) => {
            6u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Pow(l, r) => {
            7u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Neg(inner) => {
            8u8.hash(hasher);
            hash_expr_recursive(ctx, *inner, hasher);
        }
        Expr::Function(name, args) => {
            9u8.hash(hasher);
            name.hash(hasher);
            for arg in args {
                hash_expr_recursive(ctx, *arg, hasher);
            }
        }
        Expr::Matrix { rows, cols, data } => {
            10u8.hash(hasher);
            rows.hash(hasher);
            cols.hash(hasher);
            for elem in data {
                hash_expr_recursive(ctx, *elem, hasher);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_format_expr_limited_no_truncate() {
        let mut ctx = Context::new();
        let expr = parse("x + 1", &mut ctx).unwrap();
        let (s, trunc, len) = format_expr_limited(&ctx, expr, 100);
        assert!(!trunc);
        assert!(len <= 100);
        assert!(s.contains("x"));
    }

    #[test]
    fn test_format_expr_limited_truncate() {
        let mut ctx = Context::new();
        let expr = parse("x + y + z + a + b + c", &mut ctx).unwrap();
        let (s, trunc, _len) = format_expr_limited(&ctx, expr, 5);
        assert!(trunc);
        assert!(s.contains("truncated"));
    }

    #[test]
    fn test_expr_stats() {
        let mut ctx = Context::new();
        let expr = parse("x + 1", &mut ctx).unwrap();
        let stats = expr_stats(&ctx, expr);
        assert!(stats.node_count >= 3); // Add, Var, Num
        assert!(stats.depth >= 1);
    }

    #[test]
    fn test_expr_hash_deterministic() {
        let mut ctx = Context::new();
        let e1 = parse("x + 1", &mut ctx).unwrap();
        let e2 = parse("x + 1", &mut ctx).unwrap();
        assert_eq!(expr_hash(&ctx, e1), expr_hash(&ctx, e2));
    }
}
