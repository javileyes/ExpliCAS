//! Core LaTeX rendering logic shared across all LaTeX renderers
//!
//! This module contains the common rendering functions used by LaTeXExpr,
//! LaTeXExprWithHints, LaTeXExprHighlighted, and LaTeXExprHighlightedWithHints.
//! By centralizing this logic, we ensure consistent formatting across all renderers.

use crate::{Constant, Context, Expr, ExprId};
use num_rational::Rational64;
use num_traits::Signed;

/// Post-process LaTeX to fix negative sign patterns
/// Handles cases like "+ -" → "-" and "- -" → "+"
pub fn clean_latex_negatives(latex: &str) -> String {
    // Replace "+ -" with "- " (addition of negative becomes subtraction)
    let result = latex.replace("+ -", "- ");
    // Replace "- -" with "+ " (subtraction of negative becomes addition)
    let result = result.replace("- -", "+ ");
    // Handle "- \\frac" patterns that might have extra spacing
    result
}

/// Render a rational number as LaTeX
/// Negative fractions are rendered as -\frac{a}{b} instead of \frac{-a}{b}
pub fn render_number(n: &Rational64) -> String {
    if n.is_integer() {
        format!("{}", n.numer())
    } else if n.is_negative() {
        // Put negative sign outside fraction: -1/2 -> -\frac{1}{2}
        let positive = -n;
        format!("-\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
    } else {
        format!("\\frac{{{}}}{{{}}}", n.numer(), n.denom())
    }
}

/// Render a constant as LaTeX
pub fn render_constant(c: &Constant) -> String {
    match c {
        Constant::Pi => "\\pi".to_string(),
        Constant::E => "e".to_string(),
        Constant::Infinity => "\\infty".to_string(),
        Constant::Undefined => "\\text{undefined}".to_string(),
    }
}

/// Check if we should always use explicit multiplication symbol
/// Returns true to ensure consistent \cdot usage
pub fn needs_explicit_mult() -> bool {
    // Always use \cdot for consistent formatting
    true
}

/// Render a Div expression, handling negative numerators properly
pub fn render_div<F>(ctx: &Context, l: ExprId, r: ExprId, render_fn: F) -> String
where
    F: Fn(ExprId, bool) -> String,
{
    // Check if numerator is negative - put sign outside fraction
    match ctx.get(l) {
        Expr::Neg(inner) => {
            let numer = render_fn(*inner, false);
            let denom = render_fn(r, false);
            format!("-\\frac{{{}}}{{{}}}", numer, denom)
        }
        Expr::Number(n) if n.is_negative() => {
            let positive = -n;
            let numer_str = if positive.is_integer() {
                format!("{}", positive.numer())
            } else {
                format!("\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
            };
            let denom = render_fn(r, false);
            format!("-\\frac{{{}}}{{{}}}", numer_str, denom)
        }
        _ => {
            let numer = render_fn(l, false);
            let denom = render_fn(r, false);
            format!("\\frac{{{}}}{{{}}}", numer, denom)
        }
    }
}

/// Render a Mul expression with explicit \cdot
pub fn render_mul<F>(
    l_latex: String,
    r_latex: String,
    _l: ExprId,
    _r: ExprId,
    _render_mul_fn: F,
) -> String
where
    F: Fn(ExprId) -> String,
{
    // Always use cdot for consistent formatting
    format!("{} \\cdot {}", l_latex, r_latex)
}

/// Render addition, detecting negative right operand to show as subtraction
pub fn render_add<F, G>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    render_fn: F,
    render_mul_fn: G,
) -> String
where
    F: Fn(ExprId, bool) -> String,
    G: Fn(ExprId) -> String,
{
    // First check if left side is negative - canonicalization may swap operands
    if let Expr::Neg(left_inner) = ctx.get(l) {
        // Left side is negative: Add(Neg(a), b) = b - a (more natural order)
        let inner_latex = render_fn(*left_inner, true);
        let right_latex = render_fn(r, false);
        return format!("{} - {}", right_latex, inner_latex);
    }

    let left = render_fn(l, false);

    // Check if right side is a negative number, negation, or multiplication by negative
    // If so, render as subtraction instead of addition
    let (is_negative, right_str) = match ctx.get(r) {
        // Case 1: Negative number literal
        Expr::Number(n) if n.is_negative() => {
            let positive = -n;
            let positive_str = if positive.is_integer() {
                format!("{}", positive.numer())
            } else {
                format!("\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
            };
            (true, positive_str)
        }
        // Case 2: Neg(expr) - extract the inner expression
        Expr::Neg(inner) => (true, render_fn(*inner, true)),
        // Case 3: Mul with negative leading coefficient (-1 * expr, -2 * expr, etc.)
        Expr::Mul(ml, mr) => {
            // Check if left factor is a negative number
            if let Expr::Number(coef) = ctx.get(*ml) {
                if coef.is_negative() {
                    // Extract positive coefficient and rest
                    let positive_coef = -coef;
                    let rest_latex = render_mul_fn(*mr);

                    // Format as "positive_coef * rest" for subtraction
                    if positive_coef.is_integer() && *positive_coef.numer() == 1.into() {
                        // -1 * expr -> just "expr"
                        (true, rest_latex)
                    } else {
                        // -n * expr -> "n \cdot expr"
                        let coef_str = if positive_coef.is_integer() {
                            format!("{}", positive_coef.numer())
                        } else {
                            format!(
                                "\\frac{{{}}}{{{}}}",
                                positive_coef.numer(),
                                positive_coef.denom()
                            )
                        };

                        // Always use cdot for consistent formatting
                        (true, format!("{}\\cdot {}", coef_str, rest_latex))
                    }
                } else {
                    // Positive coefficient, render normally
                    (false, render_fn(r, false))
                }
            } else {
                // Left factor is not a number
                (false, render_fn(r, false))
            }
        }
        // Case 4: Regular positive expression
        _ => (false, render_fn(r, false)),
    };

    if is_negative {
        format!("{} - {}", left, right_str)
    } else {
        format!("{} + {}", left, right_str)
    }
}

/// Render subtraction
pub fn render_sub<F>(l: ExprId, r: ExprId, render_fn: F) -> String
where
    F: Fn(ExprId, bool) -> String,
{
    let left = render_fn(l, false);
    let right = render_fn(r, true);
    format!("{} - {}", left, right)
}

/// Render power expression
pub fn render_pow<F, G>(base: ExprId, exp: ExprId, render_base_fn: F, render_fn: G) -> String
where
    F: Fn(ExprId) -> String,
    G: Fn(ExprId, bool) -> String,
{
    let base_str = render_base_fn(base);
    let exp_str = render_fn(exp, false);
    format!("{{{}}}^{{{}}}", base_str, exp_str)
}

/// Render negation
pub fn render_neg<F>(e: ExprId, render_fn: F) -> String
where
    F: Fn(ExprId, bool) -> String,
{
    let inner = render_fn(e, true);
    format!("-{}", inner)
}

/// Render sum function as LaTeX \sum notation
pub fn render_sum<F>(args: &[ExprId], render_fn: F) -> String
where
    F: Fn(ExprId, bool) -> String,
{
    if args.len() == 4 {
        let expr = render_fn(args[0], false);
        let var = render_fn(args[1], false);
        let start = render_fn(args[2], false);
        let end = render_fn(args[3], false);
        format!("\\sum_{{{}={}}}^{{{}}} {}", var, start, end, expr)
    } else {
        format!(
            "\\text{{sum}}({})",
            args.iter()
                .map(|a| render_fn(*a, false))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Render product function as LaTeX \prod notation
pub fn render_product<F>(args: &[ExprId], render_fn: F) -> String
where
    F: Fn(ExprId, bool) -> String,
{
    if args.len() == 4 {
        let expr = render_fn(args[0], false);
        let var = render_fn(args[1], false);
        let start = render_fn(args[2], false);
        let end = render_fn(args[3], false);
        format!("\\prod_{{{}={}}}^{{{}}} {}", var, start, end, expr)
    } else {
        format!(
            "\\text{{product}}({})",
            args.iter()
                .map(|a| render_fn(*a, false))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Render sqrt function
pub fn render_sqrt<F>(args: &[ExprId], render_fn: F) -> String
where
    F: Fn(ExprId, bool) -> String,
{
    if args.len() == 1 {
        let arg = render_fn(args[0], false);
        format!("\\sqrt{{{}}}", arg)
    } else if args.len() == 2 {
        let radicand = render_fn(args[0], false);
        let index = render_fn(args[1], false);
        format!("\\sqrt[{}]{{{}}}", index, radicand)
    } else {
        format!(
            "\\text{{sqrt}}({})",
            args.iter()
                .map(|a| render_fn(*a, false))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Render common trig/math functions
pub fn render_trig_function<F>(name: &str, args: &[ExprId], render_fn: F) -> String
where
    F: Fn(ExprId, bool) -> String,
{
    match name {
        "sin" | "cos" | "tan" | "cot" | "sec" | "csc" | "asin" | "acos" | "atan" | "sinh"
        | "cosh" | "tanh" => {
            if args.len() == 1 {
                let arg = render_fn(args[0], false);
                format!("\\{}({})", name, arg)
            } else {
                format!(
                    "\\text{{{}}}({})",
                    name,
                    args.iter()
                        .map(|a| render_fn(*a, false))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        "ln" => {
            if args.len() == 1 {
                let arg = render_fn(args[0], false);
                format!("\\ln({})", arg)
            } else {
                format!(
                    "\\text{{ln}}({})",
                    args.iter()
                        .map(|a| render_fn(*a, false))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        "log" => {
            if args.len() == 1 {
                let arg = render_fn(args[0], false);
                format!("\\log({})", arg)
            } else if args.len() == 2 {
                let base = render_fn(args[1], false);
                let arg = render_fn(args[0], false);
                format!("\\log_{{{}}}({})", base, arg)
            } else {
                format!(
                    "\\text{{log}}({})",
                    args.iter()
                        .map(|a| render_fn(*a, false))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        "abs" => {
            if args.len() == 1 {
                let arg = render_fn(args[0], false);
                format!("|{}|", arg)
            } else {
                format!(
                    "\\text{{abs}}({})",
                    args.iter()
                        .map(|a| render_fn(*a, false))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        "exp" => {
            if args.len() == 1 {
                let arg = render_fn(args[0], false);
                format!("e^{{{}}}", arg)
            } else {
                format!(
                    "\\text{{exp}}({})",
                    args.iter()
                        .map(|a| render_fn(*a, false))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        "floor" => {
            if args.len() == 1 {
                let arg = render_fn(args[0], false);
                format!("\\lfloor {} \\rfloor", arg)
            } else {
                format!(
                    "\\text{{floor}}({})",
                    args.iter()
                        .map(|a| render_fn(*a, false))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        "ceil" => {
            if args.len() == 1 {
                let arg = render_fn(args[0], false);
                format!("\\lceil {} \\rceil", arg)
            } else {
                format!(
                    "\\text{{ceil}}({})",
                    args.iter()
                        .map(|a| render_fn(*a, false))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        "diff" => {
            if args.len() >= 2 {
                let expr = render_fn(args[0], false);
                let var = render_fn(args[1], false);
                format!("\\frac{{d}}{{d{}}}({})", var, expr)
            } else {
                format!(
                    "\\text{{diff}}({})",
                    args.iter()
                        .map(|a| render_fn(*a, false))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        "integrate" => {
            if args.len() >= 2 {
                let expr = render_fn(args[0], false);
                let var = render_fn(args[1], false);
                format!("\\int {} \\, d{}", expr, var)
            } else {
                format!(
                    "\\text{{integrate}}({})",
                    args.iter()
                        .map(|a| render_fn(*a, false))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        "matmul" => {
            if args.len() == 2 {
                let a = render_fn(args[0], false);
                let b = render_fn(args[1], false);
                format!("{} \\times {}", a, b)
            } else {
                format!(
                    "\\text{{matmul}}({})",
                    args.iter()
                        .map(|a| render_fn(*a, false))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        "transpose" => {
            if args.len() == 1 {
                let arg = render_fn(args[0], false);
                format!("{}^T", arg)
            } else {
                format!(
                    "\\text{{transpose}}({})",
                    args.iter()
                        .map(|a| render_fn(*a, false))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        _ => {
            // Generic function rendering
            let args_str = args
                .iter()
                .map(|a| render_fn(*a, false))
                .collect::<Vec<_>>()
                .join(", ");
            format!("\\text{{{}}}({})", name, args_str)
        }
    }
}
