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
    let mut display_ctx = DisplayContext::new();

    // First: scan original expression for sqrt() functions
    scan_for_sqrt_hints(ctx, original_expr, &mut display_ctx);

    // Second: capture hints from Canonicalize Roots steps
    for step in steps {
        if step.rule_name.contains("Canonicalize Roots") {
            if let Some(index) = extract_root_index(ctx, step.before) {
                display_ctx.insert(step.after, DisplayHint::AsRoot { index });
            }
        }
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
