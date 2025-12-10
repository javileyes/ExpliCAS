//! Display Context for preserving original expression forms
//!
//! This module re-exports types from cas_ast and provides the builder
//! function that analyzes Steps to construct display hints.
//!
//! # Example
//! ```ignore
//! let steps = simplifier.simplify(expr);
//! let display_ctx = build_display_context(&ctx, &steps);
//! // Use with DisplayExpr or LaTeX rendering
//! ```

use crate::step::Step;
use cas_ast::{Context, DisplayContext, DisplayHint, Expr};

/// Build DisplayContext by analyzing the original expression and simplification steps
///
/// This scans the original expression for sqrt() functions and also looks at
/// steps for "Canonicalize Roots" patterns. Zero cost if not called.
pub fn build_display_context(
    ctx: &Context,
    original_expr: cas_ast::ExprId,
    steps: &[Step],
) -> DisplayContext {
    build_display_context_with_result(ctx, original_expr, steps, None)
}

/// Build DisplayContext, optionally including the final simplified result
pub fn build_display_context_with_result(
    ctx: &Context,
    original_expr: cas_ast::ExprId,
    steps: &[Step],
    simplified_result: Option<cas_ast::ExprId>,
) -> DisplayContext {
    let mut display_ctx = DisplayContext::new();

    // Collect root patterns from original expression: (base ExprId or value, index)
    let root_patterns = collect_root_patterns(ctx, original_expr);

    // First: scan original expression for sqrt() functions
    scan_for_sqrt_hints(ctx, original_expr, &mut display_ctx);

    // Second: scan all step.before and step.after for sqrt() functions
    // This catches expressions created during simplification that are also sqrt
    for step in steps {
        scan_for_sqrt_hints(ctx, step.before, &mut display_ctx);
        scan_for_sqrt_hints(ctx, step.after, &mut display_ctx);
    }

    // Third: propagate hints to Pow expressions that match collected sqrt patterns
    // When sqrt(base, n) is converted to base^(1/n), add AsRoot hint to the Pow
    for step in steps {
        propagate_sqrt_hints_to_pow(ctx, step.before, &root_patterns, &mut display_ctx);
        propagate_sqrt_hints_to_pow(ctx, step.after, &root_patterns, &mut display_ctx);
        // Also check global_before and global_after if available
        if let Some(gb) = step.global_before {
            propagate_sqrt_hints_to_pow(ctx, gb, &root_patterns, &mut display_ctx);
        }
        if let Some(ga) = step.global_after {
            propagate_sqrt_hints_to_pow(ctx, ga, &root_patterns, &mut display_ctx);
        }
    }

    // NOTE: We do NOT auto-convert x^(1/n) to roots here.
    // If user wrote x^(1/4), show as x^(1/4). If user wrote sqrt(x), show as √.
    // Only sqrt() usage triggers root notation.

    // Fourth: capture hints from Canonicalize Roots steps
    for step in steps {
        if step.rule_name.contains("Canonicalize Roots") {
            if let Some(index) = extract_root_index(ctx, step.before) {
                display_ctx.insert(step.after, DisplayHint::AsRoot { index });
            }
        }
    }

    // Fifth: propagate hints to the simplified result (handles cases with no steps)
    if let Some(result) = simplified_result {
        propagate_sqrt_hints_to_pow(ctx, result, &root_patterns, &mut display_ctx);
    }

    display_ctx
}

/// Recursively scan expression tree for sqrt/root functions and add hints
fn scan_for_sqrt_hints(ctx: &Context, expr: cas_ast::ExprId, display_ctx: &mut DisplayContext) {
    match ctx.get(expr) {
        Expr::Function(name, args) => {
            // Check if this is a sqrt/root function
            if name == "sqrt" {
                let index = match args.len() {
                    1 => 2,
                    2 => {
                        if let Expr::Number(n) = ctx.get(args[1]) {
                            n.to_integer().try_into().unwrap_or(2)
                        } else {
                            2
                        }
                    }
                    _ => 2,
                };
                display_ctx.insert(expr, DisplayHint::AsRoot { index });
            } else if name == "root" && args.len() == 2 {
                let index = if let Expr::Number(n) = ctx.get(args[1]) {
                    n.to_integer().try_into().unwrap_or(2)
                } else {
                    2
                };
                display_ctx.insert(expr, DisplayHint::AsRoot { index });
            }
            // Recurse into arguments
            for arg in args {
                scan_for_sqrt_hints(ctx, *arg, display_ctx);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            scan_for_sqrt_hints(ctx, *l, display_ctx);
            scan_for_sqrt_hints(ctx, *r, display_ctx);
        }
        Expr::Neg(e) => {
            scan_for_sqrt_hints(ctx, *e, display_ctx);
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                scan_for_sqrt_hints(ctx, *elem, display_ctx);
            }
        }
        // Terminals: Number, Variable, Constant - no children to scan
        _ => {}
    }
}

/// Represents a root pattern collected from sqrt() functions
/// Stores index and a string representation of the base expression for matching
#[derive(Debug, Clone)]
struct RootPattern {
    index: usize,
    base_repr: String,
}

/// Collect root patterns from sqrt() functions in the expression
/// Returns a list of patterns to match against Pow expressions later
fn collect_root_patterns(ctx: &Context, expr: cas_ast::ExprId) -> Vec<RootPattern> {
    let mut patterns = Vec::new();
    collect_root_patterns_recursive(ctx, expr, &mut patterns);
    patterns
}

fn collect_root_patterns_recursive(
    ctx: &Context,
    expr: cas_ast::ExprId,
    patterns: &mut Vec<RootPattern>,
) {
    match ctx.get(expr) {
        Expr::Function(name, args) => {
            if name == "sqrt" {
                let (index, base) = match args.len() {
                    1 => (2, args[0]),
                    2 => {
                        let idx = if let Expr::Number(n) = ctx.get(args[1]) {
                            n.to_integer().try_into().unwrap_or(2)
                        } else {
                            2
                        };
                        (idx, args[0])
                    }
                    _ => (2, args[0]),
                };
                patterns.push(RootPattern {
                    index,
                    base_repr: expr_to_string(ctx, base),
                });
            }
            // Recurse into arguments
            for arg in args {
                collect_root_patterns_recursive(ctx, *arg, patterns);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            collect_root_patterns_recursive(ctx, *l, patterns);
            collect_root_patterns_recursive(ctx, *r, patterns);
        }
        Expr::Neg(e) => {
            collect_root_patterns_recursive(ctx, *e, patterns);
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                collect_root_patterns_recursive(ctx, *elem, patterns);
            }
        }
        _ => {}
    }
}

/// Compute a string representation of an expression for pattern matching
fn expr_to_string(ctx: &Context, expr: cas_ast::ExprId) -> String {
    match ctx.get(expr) {
        Expr::Number(n) => format!("N({}/{})", n.numer(), n.denom()),
        Expr::Variable(name) => format!("V({})", name),
        Expr::Constant(c) => format!("C({:?})", c),
        Expr::Add(l, r) => format!(
            "Add({},{})",
            expr_to_string(ctx, *l),
            expr_to_string(ctx, *r)
        ),
        Expr::Sub(l, r) => format!(
            "Sub({},{})",
            expr_to_string(ctx, *l),
            expr_to_string(ctx, *r)
        ),
        Expr::Mul(l, r) => format!(
            "Mul({},{})",
            expr_to_string(ctx, *l),
            expr_to_string(ctx, *r)
        ),
        Expr::Div(l, r) => format!(
            "Div({},{})",
            expr_to_string(ctx, *l),
            expr_to_string(ctx, *r)
        ),
        Expr::Pow(base, exp) => format!(
            "Pow({},{})",
            expr_to_string(ctx, *base),
            expr_to_string(ctx, *exp)
        ),
        Expr::Neg(e) => format!("Neg({})", expr_to_string(ctx, *e)),
        Expr::Function(name, args) => {
            let args_str: Vec<_> = args.iter().map(|a| expr_to_string(ctx, *a)).collect();
            format!("Fn({},{})", name, args_str.join(","))
        }
        Expr::Matrix { rows, cols, data } => {
            let data_str: Vec<_> = data.iter().map(|e| expr_to_string(ctx, *e)).collect();
            format!("Mat({},{},{})", rows, cols, data_str.join(","))
        }
    }
}

/// Propagate sqrt hints to Pow expressions that match collected patterns
fn propagate_sqrt_hints_to_pow(
    ctx: &Context,
    expr: cas_ast::ExprId,
    patterns: &[RootPattern],
    display_ctx: &mut DisplayContext,
) {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            // Check if exponent is k/n for some k and n
            if let Expr::Number(n) = ctx.get(*exp) {
                let denom: i64 = n.denom().try_into().unwrap_or(1);
                if denom > 1 {
                    // This is x^(k/n) - check if index matches any sqrt pattern
                    // We match by index only (not base) because combined roots like
                    // sqrt(3,4)*sqrt(4,4) = 12^(1/4) should still display as 4√(12)
                    for pattern in patterns {
                        if pattern.index == denom as usize {
                            display_ctx.insert(
                                expr,
                                DisplayHint::AsRoot {
                                    index: denom as u32,
                                },
                            );
                            break;
                        }
                    }
                }
            }
            // Recurse into base and exponent
            propagate_sqrt_hints_to_pow(ctx, *base, patterns, display_ctx);
            propagate_sqrt_hints_to_pow(ctx, *exp, patterns, display_ctx);
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            propagate_sqrt_hints_to_pow(ctx, *l, patterns, display_ctx);
            propagate_sqrt_hints_to_pow(ctx, *r, patterns, display_ctx);
        }
        Expr::Neg(e) => {
            propagate_sqrt_hints_to_pow(ctx, *e, patterns, display_ctx);
        }
        Expr::Function(_, args) => {
            for arg in args {
                propagate_sqrt_hints_to_pow(ctx, *arg, patterns, display_ctx);
            }
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                propagate_sqrt_hints_to_pow(ctx, *elem, patterns, display_ctx);
            }
        }
        _ => {}
    }
}

/// Recursively scan expression tree for x^(1/n) patterns and add root hints
fn scan_for_power_roots(ctx: &Context, expr: cas_ast::ExprId, display_ctx: &mut DisplayContext) {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            // Check if exponent is a fraction of form 1/n
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n.numer() == 1.into() && n.denom() > &1.into() {
                    // This is x^(1/n) - register as nth root
                    if let Ok(index) = n.denom().try_into() {
                        display_ctx.insert(expr, DisplayHint::AsRoot { index });
                    }
                }
            }
            // Recurse into base and exponent
            scan_for_power_roots(ctx, *base, display_ctx);
            scan_for_power_roots(ctx, *exp, display_ctx);
        }
        Expr::Function(_, args) => {
            for arg in args {
                scan_for_power_roots(ctx, *arg, display_ctx);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            scan_for_power_roots(ctx, *l, display_ctx);
            scan_for_power_roots(ctx, *r, display_ctx);
        }
        Expr::Neg(e) => {
            scan_for_power_roots(ctx, *e, display_ctx);
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                scan_for_power_roots(ctx, *elem, display_ctx);
            }
        }
        _ => {}
    }
}

/// Extract the root index from a sqrt/root function
fn extract_root_index(ctx: &Context, expr: cas_ast::ExprId) -> Option<u32> {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "sqrt" {
            return match args.len() {
                1 => Some(2), // sqrt(x) = 2nd root
                2 => {
                    // sqrt(x, n) = nth root
                    if let Expr::Number(n) = ctx.get(args[1]) {
                        n.to_integer().try_into().ok()
                    } else {
                        None
                    }
                }
                _ => None,
            };
        }
        if name == "root" && args.len() == 2 {
            // root(x, n) = nth root
            if let Expr::Number(n) = ctx.get(args[1]) {
                return n.to_integer().try_into().ok();
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::Step;

    #[test]
    fn test_extract_root_index_sqrt() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));

        let index = extract_root_index(&ctx, sqrt_x);
        assert_eq!(index, Some(2));
    }

    #[test]
    fn test_extract_root_index_cbrt() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let three = ctx.num(3);
        let cbrt_x = ctx.add(Expr::Function("sqrt".to_string(), vec![x, three]));

        let index = extract_root_index(&ctx, cbrt_x);
        assert_eq!(index, Some(3));
    }

    #[test]
    fn test_build_display_context() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));
        let half = ctx.rational(1, 2);
        let x_half = ctx.add(Expr::Pow(x, half));

        let step = Step::new(
            "sqrt(x) = x^(1/2)",
            "Canonicalize Roots",
            sqrt_x,
            x_half,
            vec![],
            Some(&ctx),
        );

        let display_ctx = build_display_context(&ctx, sqrt_x, &[step]);

        // Should have 2 hints: one from scanning sqrt_x, one from step
        assert!(display_ctx.len() >= 1);
        assert_eq!(
            display_ctx.get(x_half),
            Some(&DisplayHint::AsRoot { index: 2 })
        );
    }
}
