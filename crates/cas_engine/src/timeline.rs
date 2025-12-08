use crate::step::{PathStep, Step};
use cas_ast::{Context, DisplayExpr, Expr, ExprId, LaTeXExpr};
use num_traits::Signed;

/// Timeline HTML generator - exports simplification steps to interactive HTML
pub struct TimelineHtml<'a> {
    context: &'a mut Context,
    steps: &'a [Step],
    original_expr: ExprId,
    title: String,
    verbosity_level: VerbosityLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum VerbosityLevel {
    Low,     // Only high-importance steps (Factor, Expand, Integrate, etc.)
    Normal,  // Medium+ importance steps (most transformations)
    Verbose, // All steps including trivial ones
}

impl VerbosityLevel {
    /// Returns minimum importance level to show for this verbosity
    fn min_importance(&self) -> crate::step::ImportanceLevel {
        use crate::step::ImportanceLevel;
        match self {
            VerbosityLevel::Low => ImportanceLevel::High,
            VerbosityLevel::Normal => ImportanceLevel::Medium,
            VerbosityLevel::Verbose => ImportanceLevel::Trivial, // Show all
        }
    }

    /// Check if a step should be shown at this verbosity level
    fn should_show_step(&self, step: &Step) -> bool {
        step.importance() >= self.min_importance()
    }
}

impl<'a> TimelineHtml<'a> {
    pub fn new(
        context: &'a mut Context,
        steps: &'a [Step],
        original_expr: ExprId,
        verbosity: VerbosityLevel,
    ) -> Self {
        let title = format!(
            "{}",
            DisplayExpr {
                context,
                id: original_expr
            }
        );
        Self {
            context,
            steps,
            original_expr,
            title,
            verbosity_level: verbosity,
        }
    }

    /// Generate LaTeX for an expression with a specific subexpression highlighted
    /// by following the path to find the target subexpression.
    /// Uses smart context: if the target is deep in the tree, highlights a meaningful parent.
    fn generate_latex_with_highlight(
        &self,
        root_expr: ExprId,
        path: &[PathStep],
        target_expr: ExprId,
    ) -> String {
        // Strategy: If path is very deep (>3 levels) and goes into fraction denominator or
        // exponent, highlight the whole fraction/power instead of the tiny part
        let should_highlight_parent = self.should_use_parent_context(root_expr, path);

        let latex = if should_highlight_parent {
            // Find a good "stopping point" in the path (e.g., before entering fraction denominator)
            let shortened_path = self.find_highlight_context_path(root_expr, path);
            self.latex_with_highlight_recursive(root_expr, &shortened_path, 0, target_expr)
        } else {
            self.latex_with_highlight_recursive(root_expr, path, 0, target_expr)
        };

        // Phase 2: Clean up LaTeX patterns
        // This fixes issues like "+ -" and "- -" that can occur with nested negative terms
        Self::clean_latex_negatives(&latex)
    }

    /// Post-process LaTeX to fix negative sign patterns
    /// Handles cases like "+ -" → "-" and "- -" → "+"
    fn clean_latex_negatives(latex: &str) -> String {
        let mut result = latex.to_string();

        // Fix "+ -" → "-" in all contexts
        // Even "+ -(" is simplified to "-(" since +(-(x)) = -(x)
        result = result.replace("+ -\\", "- \\"); // Before LaTeX commands
        result = result.replace("+ -{", "- {"); // Before braces
        result = result.replace("+ -(", "- ("); // Before parentheses

        // Fix "- -" → "+" (double negative)
        result = result.replace("- -\\", "+ \\"); // Before LaTeX commands
        result = result.replace("- -{", "+ {"); // Before braces
        result = result.replace("- -(", "+ ("); // Before parentheses

        // Fix "+ -" before digits (e.g., "+ -4" → "- 4")
        use regex::Regex;
        let re_plus_minus_digit = Regex::new(r"\+ -(\d)").unwrap();
        result = re_plus_minus_digit.replace_all(&result, "- $1").to_string();

        // Fix "- -" before digits (e.g., "- -4" → "+ 4")
        let re_minus_minus_digit = Regex::new(r"- -(\d)").unwrap();
        result = re_minus_minus_digit
            .replace_all(&result, "+ $1")
            .to_string();

        result
    }

    /// Determine if we should highlight a parent context instead of the exact target.
    /// Only do this in truly confusing cases (deep nesting in exponents of complex expressions)
    fn should_use_parent_context(&self, root_expr: ExprId, path: &[PathStep]) -> bool {
        // If path is very short, always highlight exact target
        if path.len() <= 3 {
            return false;
        }

        // Only use parent context in very specific confusing scenarios:
        // 1. Very deep paths (>4 levels) going into exponents
        // 2. Denominators of fractions that are themselves fractions (nested fractions)

        let mut current = root_expr;
        let mut in_fraction_denominator = false;

        for (i, step) in path.iter().enumerate() {
            match (self.context.get(current), step) {
                // Track if we're entering a fraction denominator
                (Expr::Div(_, _), PathStep::Right) => {
                    // If we're already in a denominator and going deeper, that's confusing
                    if in_fraction_denominator && i > 2 {
                        return true;
                    }
                    in_fraction_denominator = true;
                }
                // Very deep exponent (>3 levels deep)
                (Expr::Pow(_, _), PathStep::Exponent) if i > 3 => return true,
                _ => {}
            }

            // Navigate to next node
            current = match (self.context.get(current), step) {
                (Expr::Add(l, _), PathStep::Left) => *l,
                (Expr::Add(_, r), PathStep::Right) => *r,
                (Expr::Sub(l, _), PathStep::Left) => *l,
                (Expr::Sub(_, r), PathStep::Right) => *r,
                (Expr::Mul(l, _), PathStep::Left) => *l,
                (Expr::Mul(_, r), PathStep::Right) => *r,
                (Expr::Div(n, _), PathStep::Left) => *n,
                (Expr::Div(_, d), PathStep::Right) => *d,
                (Expr::Pow(b, _), PathStep::Base) => *b,
                (Expr::Pow(_, e), PathStep::Exponent) => *e,
                (Expr::Neg(e), PathStep::Inner) => *e,
                (Expr::Function(_, args), PathStep::Arg(idx)) => args[*idx],
                _ => break,
            };
        }

        false
    }

    /// Find a good "stopping point" for highlighting.
    /// This should only trigger in rare, truly confusing cases.
    fn find_highlight_context_path(
        &self,
        root_expr: ExprId,
        full_path: &[PathStep],
    ) -> Vec<PathStep> {
        if full_path.len() <= 1 {
            return full_path.to_vec();
        }

        let mut current = root_expr;
        let mut in_fraction = false;

        for (i, step) in full_path.iter().enumerate() {
            match (self.context.get(current), step) {
                // If we're in a fraction denominator and it's getting complex, stop before entering
                (Expr::Div(_, _), PathStep::Right) => {
                    if in_fraction && i > 2 {
                        // Already in a fraction and going into denominator of another - stop here
                        return full_path[..i].to_vec();
                    }
                    in_fraction = true;
                }
                // For very deep exponents (>3 levels), stop before the exponent
                (Expr::Pow(_, _), PathStep::Exponent) if i > 3 => {
                    return full_path[..i].to_vec();
                }
                _ => {}
            }

            // Navigate
            current = match (self.context.get(current), step) {
                (Expr::Add(l, _), PathStep::Left) => *l,
                (Expr::Add(_, r), PathStep::Right) => *r,
                (Expr::Sub(l, _), PathStep::Left) => *l,
                (Expr::Sub(_, r), PathStep::Right) => *r,
                (Expr::Mul(l, _), PathStep::Left) => *l,
                (Expr::Mul(_, r), PathStep::Right) => *r,
                (Expr::Div(n, _), PathStep::Left) => *n,
                (Expr::Div(_, d), PathStep::Right) => *d,
                (Expr::Pow(b, _), PathStep::Base) => *b,
                (Expr::Pow(_, e), PathStep::Exponent) => *e,
                (Expr::Neg(e), PathStep::Inner) => *e,
                (Expr::Function(_, args), PathStep::Arg(idx)) => args[*idx],
                _ => break,
            };
        }

        // Default: use full path (highlight exact target)
        full_path.to_vec()
    }

    /// Recursive helper that generates LaTeX and highlights the target based on path
    fn latex_with_highlight_recursive(
        &self,
        current_expr: ExprId,
        path: &[PathStep],
        path_index: usize,
        target_expr: ExprId,
    ) -> String {
        // If we've reached the end of the path, this is the expression to highlight
        if path_index >= path.len() {
            // Wrap the target expression in red color with limited scope
            let inner_latex = LaTeXExpr {
                context: self.context,
                id: target_expr,
            }
            .to_latex();
            return format!("{{\\color{{red}}{{{}}}}}", inner_latex);
        }

        // Otherwise, continue following the path
        match self.context.get(current_expr) {
            Expr::Add(l, r) => {
                let left_latex = if matches!(path.get(path_index), Some(PathStep::Left)) {
                    self.latex_with_highlight_recursive(*l, path, path_index + 1, target_expr)
                } else {
                    LaTeXExpr {
                        context: self.context,
                        id: *l,
                    }
                    .to_latex()
                };

                let (is_negative, right_latex) = match self.context.get(*r) {
                    Expr::Number(n) if n.is_negative() => {
                        let positive = -n;
                        let positive_str = if positive.is_integer() {
                            format!("{}", positive.numer())
                        } else {
                            format!("\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
                        };
                        (true, positive_str)
                    }
                    Expr::Neg(inner) => {
                        // Always recursively generate to preserve highlighting
                        // DO NOT increment path_index - path already points to the right level
                        let inner_str = self.latex_with_highlight_recursive(
                            *inner,
                            path,
                            path_index,
                            target_expr,
                        );
                        (true, inner_str)
                    }

                    // Case: Mul with negative leading coefficient
                    Expr::Mul(ml, mr) => {
                        if let Expr::Number(coef) = self.context.get(*ml) {
                            if coef.is_negative() {
                                let positive_coef = -coef;

                                // Get the rest of the multiplication either with highlighting or normally
                                let rest_latex =
                                    if matches!(path.get(path_index), Some(PathStep::Right)) {
                                        // Highlight within the multiplication
                                        // Pass *mr (right side) not *r (the Mul itself)
                                        self.latex_with_highlight_recursive(
                                            *mr,
                                            path,
                                            path_index + 1,
                                            target_expr,
                                        )
                                    } else {
                                        // No explicit path, but still recurse to preserve potential highlighting deeper
                                        let mr_str = self.latex_with_highlight_recursive(
                                            *mr,
                                            path,
                                            path_index,
                                            target_expr,
                                        );

                                        // Add parentheses if needed
                                        if matches!(
                                            self.context.get(*mr),
                                            Expr::Add(_, _) | Expr::Sub(_, _)
                                        ) {
                                            format!("({})", mr_str)
                                        } else {
                                            mr_str
                                        }
                                    };

                                if positive_coef.is_integer() && *positive_coef.numer() == 1.into()
                                {
                                    (true, rest_latex)
                                } else {
                                    let coef_str = if positive_coef.is_integer() {
                                        format!("{}", positive_coef.numer())
                                    } else {
                                        format!(
                                            "\\frac{{{}}}{{{}}}",
                                            positive_coef.numer(),
                                            positive_coef.denom()
                                        )
                                    };

                                    let needs_cdot = matches!(
                                        (self.context.get(*ml), self.context.get(*mr)),
                                        (Expr::Number(_), Expr::Number(_))
                                            | (Expr::Number(_), Expr::Add(_, _))
                                            | (Expr::Number(_), Expr::Sub(_, _))
                                    );

                                    if needs_cdot {
                                        (true, format!("{}\\cdot {}", coef_str, rest_latex))
                                    } else {
                                        (true, format!("{}{}", coef_str, rest_latex))
                                    }
                                }
                            } else {
                                // Positive coefficient
                                let right_str =
                                    if matches!(path.get(path_index), Some(PathStep::Right)) {
                                        self.latex_with_highlight_recursive(
                                            *r,
                                            path,
                                            path_index + 1,
                                            target_expr,
                                        )
                                    } else {
                                        self.latex_with_highlight_recursive(
                                            *r,
                                            path,
                                            path_index,
                                            target_expr,
                                        )
                                    };
                                (false, right_str)
                            }
                        } else {
                            // Left factor is not a number
                            let right_str = if matches!(path.get(path_index), Some(PathStep::Right))
                            {
                                self.latex_with_highlight_recursive(
                                    *r,
                                    path,
                                    path_index + 1,
                                    target_expr,
                                )
                            } else {
                                self.latex_with_highlight_recursive(
                                    *r,
                                    path,
                                    path_index,
                                    target_expr,
                                )
                            };
                            (false, right_str)
                        }
                    }
                    _ => {
                        let right_str = if matches!(path.get(path_index), Some(PathStep::Right)) {
                            self.latex_with_highlight_recursive(
                                *r,
                                path,
                                path_index + 1,
                                target_expr,
                            )
                        } else {
                            self.latex_with_highlight_recursive(*r, path, path_index, target_expr)
                        };

                        // Check if right side is an Add whose leftmost term is negative
                        // This prevents `+ -` patterns when nested Adds have negative first terms
                        let is_neg = if let Expr::Add(nested_l, _) = self.context.get(*r) {
                            let nested_left_expr = self.context.get(*nested_l);
                            matches!(nested_left_expr, Expr::Neg(_))
                                || if let Expr::Number(n) = nested_left_expr {
                                    n.is_negative()
                                } else if let Expr::Mul(ml, _) = nested_left_expr {
                                    if let Expr::Number(n) = self.context.get(*ml) {
                                        n.is_negative()
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                }
                        } else {
                            false
                        };

                        // If the nested Add starts with a negative term, strip the leading "-"
                        // to avoid "- -" pattern when we use "-" operator
                        let right_str_clean = if is_neg && right_str.starts_with('-') {
                            right_str[1..].trim_start().to_string()
                        } else {
                            right_str
                        };

                        (is_neg, right_str_clean)
                    }
                };

                if is_negative {
                    format!("{} - {}", left_latex, right_latex)
                } else {
                    format!("{} + {}", left_latex, right_latex)
                }
            }
            Expr::Sub(l, r) => {
                let left_latex = if matches!(path.get(path_index), Some(PathStep::Left)) {
                    self.latex_with_highlight_recursive(*l, path, path_index + 1, target_expr)
                } else {
                    LaTeXExpr {
                        context: self.context,
                        id: *l,
                    }
                    .to_latex()
                };

                let right_latex = if matches!(path.get(path_index), Some(PathStep::Right)) {
                    self.latex_with_highlight_recursive(*r, path, path_index + 1, target_expr)
                } else {
                    // Need parens for subtraction
                    let r_str = LaTeXExpr {
                        context: self.context,
                        id: *r,
                    }
                    .to_latex();
                    // Add parens if needed
                    match self.context.get(*r) {
                        Expr::Add(_, _) | Expr::Sub(_, _) => format!("({})", r_str),
                        _ => r_str,
                    }
                };

                format!("{} - {}", left_latex, right_latex)
            }
            Expr::Mul(l, r) => {
                let needs_parens_left =
                    matches!(self.context.get(*l), Expr::Add(_, _) | Expr::Sub(_, _));
                let needs_parens_right =
                    matches!(self.context.get(*r), Expr::Add(_, _) | Expr::Sub(_, _));

                let mut left_latex = if matches!(path.get(path_index), Some(PathStep::Left)) {
                    self.latex_with_highlight_recursive(*l, path, path_index + 1, target_expr)
                } else {
                    LaTeXExpr {
                        context: self.context,
                        id: *l,
                    }
                    .to_latex()
                };

                let mut right_latex = if matches!(path.get(path_index), Some(PathStep::Right)) {
                    self.latex_with_highlight_recursive(*r, path, path_index + 1, target_expr)
                } else {
                    LaTeXExpr {
                        context: self.context,
                        id: *r,
                    }
                    .to_latex()
                };

                if needs_parens_left {
                    left_latex = format!("({})", left_latex);
                }
                if needs_parens_right {
                    right_latex = format!("({})", right_latex);
                }

                // Smart multiplication detection
                let needs_cdot = matches!(
                    (self.context.get(*l), self.context.get(*r)),
                    (Expr::Number(_), Expr::Number(_))
                        | (Expr::Number(_), Expr::Add(_, _))
                        | (Expr::Number(_), Expr::Sub(_, _))
                        | (Expr::Add(_, _), Expr::Number(_))
                        | (Expr::Sub(_, _), Expr::Number(_))
                );

                if needs_cdot {
                    format!("{}\\cdot {}", left_latex, right_latex)
                } else {
                    format!("{}{}", left_latex, right_latex)
                }
            }
            Expr::Div(n, d) => {
                let numer_latex = if matches!(path.get(path_index), Some(PathStep::Left)) {
                    self.latex_with_highlight_recursive(*n, path, path_index + 1, target_expr)
                } else {
                    LaTeXExpr {
                        context: self.context,
                        id: *n,
                    }
                    .to_latex()
                };

                let denom_latex = if matches!(path.get(path_index), Some(PathStep::Right)) {
                    self.latex_with_highlight_recursive(*d, path, path_index + 1, target_expr)
                } else {
                    LaTeXExpr {
                        context: self.context,
                        id: *d,
                    }
                    .to_latex()
                };

                format!("\\frac{{{}}}{{{}}}", numer_latex, denom_latex)
            }
            Expr::Pow(base, exp) => {
                let base_latex = if matches!(path.get(path_index), Some(PathStep::Base)) {
                    self.latex_with_highlight_recursive(*base, path, path_index + 1, target_expr)
                } else {
                    let base_str = LaTeXExpr {
                        context: self.context,
                        id: *base,
                    }
                    .to_latex();
                    // Add parens if needed
                    match self.context.get(*base) {
                        Expr::Add(_, _)
                        | Expr::Sub(_, _)
                        | Expr::Mul(_, _)
                        | Expr::Div(_, _)
                        | Expr::Neg(_) => format!("({})", base_str),
                        _ => base_str,
                    }
                };

                let exp_latex = if matches!(path.get(path_index), Some(PathStep::Exponent)) {
                    self.latex_with_highlight_recursive(*exp, path, path_index + 1, target_expr)
                } else {
                    LaTeXExpr {
                        context: self.context,
                        id: *exp,
                    }
                    .to_latex()
                };

                format!("{{{}}}^{{{}}}", base_latex, exp_latex)
            }
            Expr::Neg(e) => {
                let inner_latex = if matches!(path.get(path_index), Some(PathStep::Inner)) {
                    self.latex_with_highlight_recursive(*e, path, path_index + 1, target_expr)
                } else {
                    // Add parens if needed
                    let e_str = LaTeXExpr {
                        context: self.context,
                        id: *e,
                    }
                    .to_latex();
                    match self.context.get(*e) {
                        Expr::Add(_, _) | Expr::Sub(_, _) => format!("({})", e_str),
                        _ => e_str,
                    }
                };

                format!("-{}", inner_latex)
            }
            Expr::Function(name, args) => {
                // Check if we need to highlight a specific argument
                let highlighted_args: Vec<String> = args
                    .iter()
                    .enumerate()
                    .map(|(i, &arg)| {
                        if matches!(path.get(path_index), Some(PathStep::Arg(idx)) if *idx == i) {
                            self.latex_with_highlight_recursive(
                                arg,
                                path,
                                path_index + 1,
                                target_expr,
                            )
                        } else {
                            LaTeXExpr {
                                context: self.context,
                                id: arg,
                            }
                            .to_latex()
                        }
                    })
                    .collect();

                match name.as_str() {
                    "sqrt" if highlighted_args.len() == 1 => {
                        format!("\\sqrt{{{}}}", highlighted_args[0])
                    }
                    "sqrt" if highlighted_args.len() == 2 => {
                        format!("\\sqrt[{}]{{{}}}", highlighted_args[1], highlighted_args[0])
                    }
                    "sin" | "cos" | "tan" | "cot" | "sec" | "csc" => {
                        format!("\\{}({})", name, highlighted_args[0])
                    }
                    "ln" => {
                        format!("\\ln({})", highlighted_args[0])
                    }
                    "log" if highlighted_args.len() == 2 => {
                        // Check if base (args[0]) is constant e - if so, use ln
                        let base_arg = args[0];
                        if let Expr::Constant(cas_ast::Constant::E) = self.context.get(base_arg) {
                            format!("\\ln({})", highlighted_args[1])
                        } else {
                            format!("\\log_{{{}}}({})", highlighted_args[0], highlighted_args[1])
                        }
                    }
                    "abs" if highlighted_args.len() == 1 => {
                        format!("|{}|", highlighted_args[0])
                    }
                    // Matrix product: matmul(A, B) → A \times B
                    "matmul" if highlighted_args.len() == 2 => {
                        format!("{} \\times {}", highlighted_args[0], highlighted_args[1])
                    }
                    // Matrix transpose: transpose(A) → A^T (with parens if needed)
                    "transpose" | "T" if highlighted_args.len() == 1 => {
                        // Check if arg needs parens (matmul, etc.)
                        let needs_parens = matches!(
                            self.context.get(args[0]),
                            Expr::Add(_, _)
                                | Expr::Sub(_, _)
                                | Expr::Mul(_, _)
                                | Expr::Div(_, _)
                                | Expr::Function(_, _)
                        );
                        if needs_parens {
                            format!("({})^{{T}}", highlighted_args[0])
                        } else {
                            format!("{}^{{T}}", highlighted_args[0])
                        }
                    }
                    _ => {
                        format!("\\text{{{}}}({})", name, highlighted_args.join(", "))
                    }
                }
            }
            // Leaf nodes - should not be reached if path is valid
            _ => LaTeXExpr {
                context: self.context,
                id: current_expr,
            }
            .to_latex(),
        }
    }

    /// Generate complete HTML document
    pub fn to_html(&mut self) -> String {
        // Filter steps based on verbosity level
        let filtered_steps: Vec<&Step> = self
            .steps
            .iter()
            .filter(|step| self.verbosity_level.should_show_step(step))
            .collect();

        let mut html = Self::html_header(&self.title);
        html.push_str(&self.render_timeline_filtered(&filtered_steps));
        html.push_str(Self::html_footer());
        html
    }

    fn html_header(title: &str) -> String {
        let escaped_title = html_escape(title);
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CAS Steps: {}</title>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 12px;
            padding: 30px 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .original {{
            text-align: center;
            font-size: 1.2em;
            padding: 15px;
            background: #f0f4ff;
            border-radius: 8px;
            margin-bottom: 30px;
            border: 2px solid #667eea;
        }}
        .timeline {{
            position: relative;
            padding: 20px 0;
        }}
        .timeline::before {{
            content: '';
            position: absolute;
            left: 30px;
            top: 0;
            bottom: 0;
            width: 3px;
            background: linear-gradient(to bottom, #667eea, #764ba2);
        }}
        .step {{
            position: relative;
            margin-bottom: 30px;
            padding-left: 80px;
            animation: fadeIn 0.5s ease-in;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .step-number {{
            position: absolute;
            left: 0;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 20px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        .step-content {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .step-content:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }}
        .step-content h3 {{
            margin-top: 0;
            color: #667eea;
            font-size: 1.1em;
        }}
        .math-expr {{
            padding: 12px 15px;
            background: #fafafa;
            border-left: 4px solid #667eea;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 1.05em;
        }}
        .math-expr.before {{
            border-left-color: #ff9800;
            background: #fff8f0;
        }}
        .math-expr.after {{
            border-left-color: #4caf50;
            background: #f0fff4;
        }}
        .math-expr strong {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .rule-description {{
            text-align: center;
            padding: 12px 20px;
            margin: 15px 0;
            background: #f9f5ff;
            border-radius: 6px;
            font-size: 0.95em;
            color: #667eea;
            border: 2px dashed #667eea;
        }}
        .local-change {{
            font-size: 1.1em;
            margin: 8px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
            text-align: center;
        }}
        .rule-name {{
            font-size: 0.85em;
            color: #764ba2;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .final-result {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
            color: white;
            border-radius: 10px;
            margin-top: 30px;
            font-size: 1.2em;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        }}
        footer {{
            text-align: center;
            margin-top: 30px;
            color: white;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CAS Simplification Steps</h1>
        <p class="subtitle">Step-by-step visualization</p>
        <div class="original">
            \(\textbf{{Original Expression:}}\)
            \[{}\]
        </div>
"#,
            escaped_title,
            latex_escape(title)
        )
    }

    fn reconstruct_global_expr(
        &mut self,
        root: ExprId,
        path: &[PathStep],
        replacement: ExprId,
    ) -> ExprId {
        if path.is_empty() {
            return replacement;
        }

        let current_step = &path[0];
        let remaining_path = &path[1..];
        let expr = self.context.get(root).clone();

        match (expr, current_step) {
            (Expr::Add(l, r), PathStep::Left) => {
                let new_l = self.reconstruct_global_expr(l, remaining_path, replacement);
                self.context.add(Expr::Add(new_l, r))
            }
            (Expr::Add(l, r), PathStep::Right) => {
                let new_r = self.reconstruct_global_expr(r, remaining_path, replacement);
                self.context.add(Expr::Add(l, new_r))
            }
            (Expr::Sub(l, r), PathStep::Left) => {
                let new_l = self.reconstruct_global_expr(l, remaining_path, replacement);
                self.context.add(Expr::Sub(new_l, r))
            }
            (Expr::Sub(l, r), PathStep::Right) => {
                let new_r = self.reconstruct_global_expr(r, remaining_path, replacement);
                self.context.add(Expr::Sub(l, new_r))
            }
            (Expr::Mul(l, r), PathStep::Left) => {
                let new_l = self.reconstruct_global_expr(l, remaining_path, replacement);
                self.context.add(Expr::Mul(new_l, r))
            }
            (Expr::Mul(l, r), PathStep::Right) => {
                let new_r = self.reconstruct_global_expr(r, remaining_path, replacement);
                self.context.add(Expr::Mul(l, new_r))
            }
            (Expr::Div(l, r), PathStep::Left) => {
                let new_l = self.reconstruct_global_expr(l, remaining_path, replacement);
                self.context.add(Expr::Div(new_l, r))
            }
            (Expr::Div(l, r), PathStep::Right) => {
                let new_r = self.reconstruct_global_expr(r, remaining_path, replacement);
                self.context.add(Expr::Div(l, new_r))
            }
            (Expr::Pow(b, e), PathStep::Base) => {
                let new_b = self.reconstruct_global_expr(b, remaining_path, replacement);
                self.context.add(Expr::Pow(new_b, e))
            }
            (Expr::Pow(b, e), PathStep::Exponent) => {
                let new_e = self.reconstruct_global_expr(e, remaining_path, replacement);
                self.context.add(Expr::Pow(b, new_e))
            }
            (Expr::Neg(e), PathStep::Inner) => {
                let new_e = self.reconstruct_global_expr(e, remaining_path, replacement);
                self.context.add(Expr::Neg(new_e))
            }
            (Expr::Function(name, args), PathStep::Arg(idx)) => {
                let mut new_args = args.clone();
                if *idx < new_args.len() {
                    new_args[*idx] =
                        self.reconstruct_global_expr(new_args[*idx], remaining_path, replacement);
                    self.context.add(Expr::Function(name, new_args))
                } else {
                    root
                }
            }
            _ => root,
        }
    }

    fn render_timeline_filtered(&mut self, filtered_steps: &[&Step]) -> String {
        let mut html = String::from("        <div class=\"timeline\">\n");

        let mut step_number = 0;
        let mut current_global = self.original_expr;

        // Track which steps to display (create a set of their indices)
        let filtered_indices: std::collections::HashSet<_> = filtered_steps
            .iter()
            .map(|s| *s as *const Step) // Dereference `&Step` to `Step` before taking pointer
            .collect();

        // Iterate over ALL steps to correctly update the global state
        for step in self.steps.iter() {
            // Store global state BEFORE this step is applied
            let global_state_before_this_step = current_global;

            // Always update global state, regardless of whether it's filtered
            current_global = self.reconstruct_global_expr(current_global, &step.path, step.after);

            // Only render if this step is in filtered set
            let step_ptr = step as *const Step;
            if !filtered_indices.contains(&step_ptr) {
                continue; // Skip rendering for non-filtered steps
            }
            step_number += 1;

            // Global state BEFORE this step (this is the state *before* the current `step` was applied)
            // Highlight the part that will change
            let global_before = self.generate_latex_with_highlight(
                global_state_before_this_step,
                &step.path,
                step.before,
            );

            // Global AFTER is the current_global (already updated)
            let _global_after = LaTeXExpr {
                context: self.context,
                id: current_global,
            }
            .to_latex();

            // Local change (before -> after)
            // For certain rules, preserve exponent notation instead of converting to roots
            let should_preserve_exponents = step.rule_name.contains("Multiply exponents")
                || step.rule_name.contains("Power of a Power");

            let local_change_latex = if should_preserve_exponents {
                let local_before = cas_ast::LatexNoRoots {
                    context: self.context,
                    id: step.before,
                }
                .to_latex();
                let local_after = cas_ast::LatexNoRoots {
                    context: self.context,
                    id: step.after,
                }
                .to_latex();
                format!("{} \\rightarrow {}", local_before, local_after)
            } else {
                let local_before = LaTeXExpr {
                    context: self.context,
                    id: step.before,
                }
                .to_latex();
                let local_after = LaTeXExpr {
                    context: self.context,
                    id: step.after,
                }
                .to_latex();
                format!("{} \\rightarrow {}", local_before, local_after)
            };

            html.push_str(&format!(
                r#"            <div class="step">
                <div class="step-number">{}</div>
                <div class="step-content">
                    <h3>{}</h3>
                    <div class="math-expr before">
                        \(\textbf{{Expression:}}\)
                        \[{}\]
                    </div>
                    <div class="rule-description">
                        <div class="rule-name">\(\text{{{}}}\)</div>
                        <div class="local-change">
                            \[{}\]
                        </div>
                    </div>
                </div>
            </div>
"#,
                step_number,
                html_escape(&step.rule_name),
                global_before,
                step.description,
                local_change_latex
            ));
        }

        // Add final result
        let final_expr = LaTeXExpr {
            context: self.context,
            id: current_global,
        }
        .to_latex();
        html.push_str(
            r#"        </div>
        <div class="final-result">
            \(\textbf{Final Result:}\)
            \["#,
        );
        html.push_str(&final_expr);
        html.push_str(
            r#"\]
        </div>
    </div>
"#,
        );
        html
    }

    fn html_footer() -> &'static str {
        r#"    <footer>
        Generated by Rust CAS Engine
    </footer>
</body>
</html>"#
    }
}

/// Escape HTML special characters
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Prepare string for LaTeX rendering in MathJax
fn latex_escape(s: &str) -> String {
    // For MathJax, we mostly just need the string as-is
    // But escape backslashes that aren't part of LaTeX commands
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_generation() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let three = ctx.num(3);
        let add_expr = ctx.add(Expr::Add(two, three));
        let five = ctx.num(5);

        // Create a step for the simplification
        let steps = vec![Step::new(
            "2 + 3 = 5",
            "Combine Constants",
            add_expr,
            five,
            vec![],
            Some(&ctx),
        )];

        let mut timeline = TimelineHtml::new(&mut ctx, &steps, add_expr, VerbosityLevel::Verbose);
        let html = timeline.to_html();

        assert!(html.contains("<!DOCTYPE html"));
        assert!(html.contains("timeline"));
        assert!(html.contains("CAS Simplification"));
        // The HTML should contain our step (Combine Constants has ImportanceLevel::Low, so needs Verbose)
        assert!(html.contains("Combine Constants"));
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("x & y"), "x &amp; y");
    }
}
