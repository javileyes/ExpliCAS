//! Shared display table for simple (head-applied) function calls.
//!
//! Every LaTeX renderer in this crate — the `LaTeXRenderer` trait family and
//! the path-tracking `PathHighlightedLatexRenderer` — must produce the SAME
//! head for the same function name, or the didactic wire ends up with a
//! `rule_latex` fragment (`\arctan`, `\cosh`, `e^{..}`) that never appears in
//! the global `before_latex`/`after_latex` render (`\text{arctan}`, …). This
//! table is that single source of truth; renderers keep only the structural
//! forms (sqrt, diff, integrate, matrices, …) that need renderer-specific
//! child handling.

/// Render a simple function call over its already-rendered first argument.
///
/// Returns `None` for names (or arities) that need structural handling; the
/// caller falls through to its own match. Callers must only pass `arg0` for
/// non-empty argument lists.
pub(crate) fn unary_function_latex(name: &str, arg_count: usize, arg0: &str) -> Option<String> {
    let rendered = match name {
        "sin" | "cos" | "tan" | "cot" | "sec" | "csc" => format!("\\{}({})", name, arg0),
        // Inverse trig: \arcsin, \arccos, \arctan (MathJax-compatible)
        "asin" | "arcsin" => format!("\\arcsin({})", arg0),
        "acos" | "arccos" => format!("\\arccos({})", arg0),
        "atan" | "arctan" => format!("\\arctan({})", arg0),
        "sinh" | "cosh" | "tanh" => format!("\\{}({})", name, arg0),
        // Inverse hyperbolic and matrix functions: proper math operators
        // (\operatorname) instead of the \text{name}(...) fallback.
        "asinh" | "acosh" | "atanh" | "asech" | "acsch" | "acoth" if arg_count == 1 => {
            format!("\\operatorname{{{}}}({})", name, arg0)
        }
        "det" | "trace" | "transpose" | "inverse" | "adjugate" | "rref" | "charpoly"
        | "eigenvalues" | "eigenvectors" | "rank" | "nullspace"
            if arg_count == 1 =>
        {
            format!("\\operatorname{{{}}}({})", name, arg0)
        }
        "ln" => format!("\\ln({})", arg0),
        "log" if arg_count == 1 => format!("\\log({})", arg0),
        "log10" if arg_count == 1 => format!("\\log_{{10}}({})", arg0),
        "abs" => format!("|{}|", arg0),
        "exp" => format!("e^{{{}}}", arg0),
        "floor" => format!("\\lfloor {} \\rfloor", arg0),
        "ceil" => format!("\\lceil {} \\rceil", arg0),
        _ => return None,
    };
    Some(rendered)
}
