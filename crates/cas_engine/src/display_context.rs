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

/// Build DisplayContext by analyzing simplification steps
///
/// This scans through steps looking for patterns like "Canonicalize Roots"
/// and records hints for proper display. Zero cost if not called.
pub fn build_display_context(ctx: &Context, steps: &[Step]) -> DisplayContext {
    let mut display_ctx = DisplayContext::new();

    for step in steps {
        // Detect "Canonicalize Roots" rule: sqrt(x) → x^(1/2)
        if step.rule_name.contains("Canonicalize Roots") {
            if let Some(index) = extract_root_index(ctx, step.before) {
                display_ctx.insert(step.after, DisplayHint::AsRoot { index });
            }
        }
    }

    display_ctx
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

        let display_ctx = build_display_context(&ctx, &[step]);

        assert_eq!(display_ctx.len(), 1);
        assert_eq!(
            display_ctx.get(x_half),
            Some(&DisplayHint::AsRoot { index: 2 })
        );
    }
}
