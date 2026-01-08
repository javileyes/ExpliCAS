use crate::step::{PathStep, Step};
use cas_ast::{
    Context, DisplayExpr, Expr, ExprId, HighlightColor, HighlightConfig, LaTeXExpr,
    LaTeXExprHighlighted,
};
use num_traits::Signed;

/// Timeline HTML generator - exports simplification steps to interactive HTML
pub struct TimelineHtml<'a> {
    context: &'a mut Context,
    steps: &'a [Step],
    original_expr: ExprId,
    simplified_result: Option<ExprId>, // Optional: the final simplified result
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
    /// Check if a step should be shown at this verbosity level
    /// Uses step.get_importance() as the single source of truth
    fn should_show_step(&self, step: &Step) -> bool {
        use crate::step::ImportanceLevel;

        match self {
            VerbosityLevel::Verbose => true,
            VerbosityLevel::Low => step.get_importance() >= ImportanceLevel::High,
            VerbosityLevel::Normal => step.get_importance() >= ImportanceLevel::Medium,
        }
    }
}

impl<'a> TimelineHtml<'a> {
    pub fn new(
        context: &'a mut Context,
        steps: &'a [Step],
        original_expr: ExprId,
        verbosity: VerbosityLevel,
    ) -> Self {
        Self::new_with_result(context, steps, original_expr, None, verbosity)
    }

    /// Create a new TimelineHtml with a known simplified result
    pub fn new_with_result(
        context: &'a mut Context,
        steps: &'a [Step],
        original_expr: ExprId,
        simplified_result: Option<ExprId>,
        verbosity: VerbosityLevel,
    ) -> Self {
        let title = LaTeXExpr {
            context,
            id: original_expr,
        }
        .to_latex();
        Self {
            context,
            steps,
            original_expr,
            simplified_result,
            title,
            verbosity_level: verbosity,
        }
    }

    // ========================================================================
    // Legacy LaTeX generation methods - kept for potential future use
    // The new trait-based LaTeXRenderer in cas_ast::latex_core provides the
    // main rendering logic. These methods are preserved for alternative
    // rendering strategies that may be needed.
    // ========================================================================

    #[allow(dead_code)]
    /// Generate LaTeX for an expression with a single subexpression highlighted
    /// Uses the new LaTeXExprHighlighted system for cleaner code.
    fn latex_with_single_highlight(
        &self,
        root_expr: ExprId,
        highlight_id: ExprId,
        color: HighlightColor,
    ) -> String {
        let mut config = HighlightConfig::new();
        config.add(highlight_id, color);
        LaTeXExprHighlighted {
            context: self.context,
            id: root_expr,
            highlights: &config,
        }
        .to_latex()
    }

    #[allow(dead_code)]
    /// Generate LaTeX for an expression with a specific subexpression highlighted
    /// by following the path to find the target subexpression.
    /// Uses smart context: if the target is deep in the tree, highlights a meaningful parent.
    fn generate_latex_with_highlight(
        &self,
        root_expr: ExprId,
        path: &[PathStep],
        _target_expr: ExprId,
    ) -> String {
        // Follow the path to find the actual expression to highlight in root_expr
        // This is needed because step.before may have different ExprId than what's in root_expr
        let actual_target = self.find_expr_at_path(root_expr, path);

        // Use LaTeXExprHighlighted which correctly handles negation rendering
        let mut config = HighlightConfig::new();
        config.add(actual_target, HighlightColor::Red);
        LaTeXExprHighlighted {
            context: self.context,
            id: root_expr,
            highlights: &config,
        }
        .to_latex()
    }

    #[allow(dead_code)]
    /// Follow a path to find the expression at that location
    fn find_expr_at_path(&self, root: ExprId, path: &[PathStep]) -> ExprId {
        let mut current = root;
        for step in path.iter() {
            let expr = self.context.get(current);
            current = match (expr, step) {
                (Expr::Add(l, _), PathStep::Left) => {
                    // Handle case where left is Neg
                    if let Expr::Neg(inner) = self.context.get(*l) {
                        *inner
                    } else {
                        *l
                    }
                }
                (Expr::Add(_, r), PathStep::Right) => {
                    // Handle case where right is Neg
                    if let Expr::Neg(inner) = self.context.get(*r) {
                        *inner
                    } else {
                        *r
                    }
                }
                (Expr::Sub(l, _), PathStep::Left) => *l,
                (Expr::Sub(_, r), PathStep::Right) => *r,
                (Expr::Mul(l, _), PathStep::Left) => *l,
                (Expr::Mul(_, r), PathStep::Right) => *r,
                (Expr::Div(l, _), PathStep::Left) => *l,
                (Expr::Div(_, r), PathStep::Right) => *r,
                (Expr::Pow(b, _), PathStep::Base) => *b,
                (Expr::Pow(_, e), PathStep::Exponent) => *e,
                (Expr::Neg(e), PathStep::Inner) => *e,
                (Expr::Function(_, args), PathStep::Arg(idx)) => {
                    if *idx < args.len() {
                        args[*idx]
                    } else {
                        break;
                    }
                }
                _ => break,
            };
        }
        current
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
                        // Handle Add(l, Neg(inner)) rendered as l - inner
                        // If path points Right, consume it before entering inner
                        let adjusted_idx = if matches!(path.get(path_index), Some(PathStep::Right))
                        {
                            // Path points to the Neg wrapper, skip to inner
                            // But also check for PathStep::Inner which would be next
                            if matches!(path.get(path_index + 1), Some(PathStep::Inner)) {
                                path_index + 2 // Consume both Right and Inner
                            } else {
                                path_index + 1 // Just consume Right
                            }
                        } else {
                            path_index // Path doesn't point here
                        };

                        let inner_str = self.latex_with_highlight_recursive(
                            *inner,
                            path,
                            adjusted_idx,
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

        // Enrich steps with didactic sub-steps
        let enriched_steps =
            crate::didactic::enrich_steps(self.context, self.original_expr, self.steps.to_vec());

        let mut html = Self::html_header(&self.title);
        html.push_str(&self.render_timeline_filtered_enriched(&filtered_steps, &enriched_steps));
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
        /* Theme variables */
        :root {{
            --bg-gradient-start: #1a1a2e;
            --bg-gradient-end: #16213e;
            --container-bg: rgba(30, 40, 60, 0.95);
            --container-shadow: rgba(0, 0, 0, 0.3);
            --title-color: #64b5f6;
            --subtitle-color: #90caf9;
            --original-bg-start: #1565c0;
            --original-bg-end: #0d47a1;
            --original-shadow: rgba(21, 101, 192, 0.4);
            --timeline-line-start: #64b5f6;
            --timeline-line-end: #4caf50;
            --step-bg: rgba(40, 50, 70, 0.8);
            --step-border: #64b5f6;
            --step-hover-shadow: rgba(100, 181, 246, 0.3);
            --step-dot-bg: #64b5f6;
            --step-dot-border: #1a1a2e;
            --step-number-color: #64b5f6;
            --content-bg: rgba(30, 40, 55, 0.9);
            --content-border: rgba(100, 181, 246, 0.2);
            --content-h3-color: #90caf9;
            --math-bg: rgba(30, 40, 55, 0.8);
            --math-border: #64b5f6;
            --math-before-border: #ff9800;
            --math-before-bg: rgba(255, 152, 0, 0.1);
            --math-after-border: #4caf50;
            --math-after-bg: rgba(76, 175, 80, 0.1);
            --math-strong-color: #b0bec5;
            --rule-bg: rgba(100, 181, 246, 0.1);
            --rule-color: #90caf9;
            --rule-border: rgba(100, 181, 246, 0.4);
            --rule-name-color: #bb86fc;
            --local-change-bg: rgba(30, 40, 55, 0.8);
            --final-bg-start: #2e7d32;
            --final-bg-end: #1b5e20;
            --final-shadow: rgba(76, 175, 80, 0.3);
            --footer-color: #90caf9;
            --substeps-bg: rgba(255, 152, 0, 0.1);
            --substeps-border: rgba(255, 152, 0, 0.3);
            --substeps-summary-color: #ffb74d;
            --substeps-summary-hover: #ffa726;
            --substeps-content-bg: rgba(30, 40, 55, 0.9);
            --substep-border: rgba(255, 152, 0, 0.2);
            --substep-desc-color: #b0bec5;
            --substep-math-bg: rgba(30, 40, 55, 0.8);
            --warning-bg: rgba(255, 193, 7, 0.15);
            --warning-border: rgba(255, 193, 7, 0.4);
            --warning-color: #ffd54f;
            --text-color: #e0e0e0;
        }}
        :root.light {{
            --bg-gradient-start: #667eea;
            --bg-gradient-end: #764ba2;
            --container-bg: white;
            --container-shadow: rgba(0, 0, 0, 0.2);
            --title-color: #333;
            --subtitle-color: #666;
            --original-bg-start: #f0f4ff;
            --original-bg-end: #f0f4ff;
            --original-shadow: rgba(102, 126, 234, 0.2);
            --timeline-line-start: #667eea;
            --timeline-line-end: #764ba2;
            --step-bg: white;
            --step-border: #667eea;
            --step-hover-shadow: rgba(102, 126, 234, 0.3);
            --step-dot-bg: #667eea;
            --step-dot-border: white;
            --step-number-color: #667eea;
            --content-bg: #fafafa;
            --content-border: #e0e0e0;
            --content-h3-color: #667eea;
            --math-bg: #fafafa;
            --math-border: #667eea;
            --math-before-border: #ff9800;
            --math-before-bg: #fff8f0;
            --math-after-border: #4caf50;
            --math-after-bg: #f0fff4;
            --math-strong-color: #666;
            --rule-bg: #f9f5ff;
            --rule-color: #667eea;
            --rule-border: #667eea;
            --rule-name-color: #764ba2;
            --local-change-bg: white;
            --final-bg-start: #4caf50;
            --final-bg-end: #45a049;
            --final-shadow: rgba(76, 175, 80, 0.3);
            --footer-color: white;
            --substeps-bg: #fff8e1;
            --substeps-border: #ffcc80;
            --substeps-summary-color: #ef6c00;
            --substeps-summary-hover: #e65100;
            --substeps-content-bg: white;
            --substep-border: #ffe0b2;
            --substep-desc-color: #795548;
            --substep-math-bg: #fafafa;
            --warning-bg: #fff3cd;
            --warning-border: #ffc107;
            --warning-color: #856404;
            --text-color: #333;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px 15px;
            background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
            min-height: 100vh;
            color: var(--text-color);
            transition: background 0.3s ease;
        }}
        .container {{
            background: var(--container-bg);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px var(--container-shadow);
            transition: background 0.3s ease;
        }}
        /* Theme toggle switch */
        .theme-toggle {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 8px;
            background: var(--container-bg);
            padding: 8px 12px;
            border-radius: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        .theme-toggle span {{
            font-size: 1.2em;
        }}
        .toggle-switch {{
            position: relative;
            width: 50px;
            height: 26px;
        }}
        .toggle-switch input {{
            opacity: 0;
            width: 0;
            height: 0;
        }}
        .toggle-slider {{
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #333;
            transition: 0.3s;
            border-radius: 26px;
        }}
        .toggle-slider:before {{
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: 0.3s;
            border-radius: 50%;
        }}
        input:checked + .toggle-slider {{
            background-color: #64b5f6;
        }}
        input:checked + .toggle-slider:before {{
            transform: translateX(24px);
        }}
        h1 {{
            color: var(--title-color);
            text-align: center;
            margin-bottom: 10px;
            font-size: 1.8em;
            transition: color 0.3s ease;
        }}
        .subtitle {{
            text-align: center;
            color: var(--subtitle-color);
            margin-bottom: 25px;
            transition: color 0.3s ease;
        }}
        .original {{
            background: linear-gradient(135deg, var(--original-bg-start), var(--original-bg-end));
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 15px var(--original-shadow);
            transition: background 0.3s ease;
        }}
        .timeline {{
            position: relative;
            padding-left: 30px;
        }}
        .timeline::before {{
            content: '';
            position: absolute;
            left: 10px;
            top: 0;
            bottom: 0;
            width: 3px;
            background: linear-gradient(to bottom, var(--timeline-line-start), var(--timeline-line-end));
            transition: background 0.3s ease;
        }}
        .step {{
            background: var(--step-bg);
            border-radius: 10px;
            padding: 15px 20px;
            margin-bottom: 20px;
            position: relative;
            border-left: 4px solid var(--step-border);
            transition: transform 0.2s, box-shadow 0.2s, background 0.3s ease;
        }}
        .step:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 20px var(--step-hover-shadow);
        }}
        .step::before {{
            content: '';
            position: absolute;
            left: -23px;
            top: 20px;
            width: 12px;
            height: 12px;
            background: var(--step-dot-bg);
            border-radius: 50%;
            border: 3px solid var(--step-dot-border);
            transition: background 0.3s ease;
        }}
        .step-number {{
            color: var(--step-number-color);
            font-weight: bold;
            font-size: 0.9em;
            margin-bottom: 5px;
            transition: color 0.3s ease;
        }}
        .step-content {{
            background: var(--content-bg);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid var(--content-border);
            transition: transform 0.2s, box-shadow 0.2s, background 0.3s ease;
        }}
        .step-content:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 16px var(--step-hover-shadow);
        }}
        .step-content h3 {{
            margin-top: 0;
            color: var(--content-h3-color);
            font-size: 1.1em;
            transition: color 0.3s ease;
        }}
        .math-expr {{
            padding: 12px 15px;
            background: var(--math-bg);
            border-left: 4px solid var(--math-border);
            margin: 10px 0;
            border-radius: 4px;
            font-size: 1.05em;
            transition: background 0.3s ease;
        }}
        .math-expr.before {{
            border-left-color: var(--math-before-border);
            background: var(--math-before-bg);
        }}
        .math-expr.after {{
            border-left-color: var(--math-after-border);
            background: var(--math-after-bg);
        }}
        .math-expr strong {{
            color: var(--math-strong-color);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            transition: color 0.3s ease;
        }}
        .rule-description {{
            text-align: center;
            padding: 12px 20px;
            margin: 15px 0;
            background: var(--rule-bg);
            border-radius: 6px;
            font-size: 0.95em;
            color: var(--rule-color);
            border: 2px dashed var(--rule-border);
            transition: background 0.3s ease, color 0.3s ease;
        }}
        .local-change {{
            font-size: 1.1em;
            margin: 8px 0;
            padding: 10px;
            background: var(--local-change-bg);
            border-radius: 4px;
            text-align: center;
            transition: background 0.3s ease;
        }}
        .rule-name {{
            font-size: 0.85em;
            color: var(--rule-name-color);
            font-weight: bold;
            margin-bottom: 5px;
            transition: color 0.3s ease;
        }}
        .final-result {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, var(--final-bg-start), var(--final-bg-end));
            color: white;
            border-radius: 10px;
            margin-top: 30px;
            font-size: 1.2em;
            box-shadow: 0 4px 12px var(--final-shadow);
            transition: background 0.3s ease;
        }}
        footer {{
            text-align: center;
            margin-top: 30px;
            color: var(--footer-color);
            font-size: 0.9em;
            transition: color 0.3s ease;
        }}
        /* Expandable details for didactic sub-steps */
        .substeps-details {{
            margin: 10px 0;
            padding: 10px 15px;
            background: var(--substeps-bg);
            border: 1px solid var(--substeps-border);
            border-radius: 8px;
            font-size: 0.95em;
            transition: background 0.3s ease;
        }}
        .substeps-details summary {{
            cursor: pointer;
            font-weight: bold;
            color: var(--substeps-summary-color);
            padding: 5px 0;
            transition: color 0.3s ease;
        }}
        .substeps-details summary:hover {{
            color: var(--substeps-summary-hover);
        }}
        .substeps-content {{
            margin-top: 10px;
            padding: 10px;
            background: var(--substeps-content-bg);
            border-radius: 6px;
            transition: background 0.3s ease;
        }}
        .substep {{
            padding: 8px 0;
            border-bottom: 1px dashed var(--substep-border);
        }}
        .substep:last-child {{
            border-bottom: none;
        }}
        .substep-desc {{
            font-weight: 500;
            color: var(--substep-desc-color);
            display: block;
            margin-bottom: 5px;
            transition: color 0.3s ease;
        }}
        .substep-math {{
            padding: 5px 10px;
            background: var(--substep-math-bg);
            border-radius: 4px;
            text-align: center;
            transition: background 0.3s ease;
        }}
        .domain-warning {{
            margin-top: 10px;
            padding: 8px 12px;
            background: var(--warning-bg);
            border: 1px solid var(--warning-border);
            border-radius: 6px;
            color: var(--warning-color);
            font-size: 0.9em;
            transition: background 0.3s ease, color 0.3s ease;
        }}
        .domain-warning::before {{
            content: '⚠ ';
        }}
        .domain-requires {{
            margin-top: 10px;
            padding: 8px 12px;
            background: rgba(33, 150, 243, 0.15);
            border: 1px solid rgba(33, 150, 243, 0.4);
            border-radius: 6px;
            color: #64b5f6;
            font-size: 0.9em;
            transition: background 0.3s ease, color 0.3s ease;
        }}
        .domain-requires::before {{
            content: 'ℹ️ ';
        }}
    </style>
</head>
<body>
    <div class="theme-toggle">
        <span>🌙</span>
        <label class="toggle-switch">
            <input type="checkbox" id="themeToggle" onchange="toggleTheme()">
            <span class="toggle-slider"></span>
        </label>
        <span>☀️</span>
    </div>
    <script>
        function toggleTheme() {{
            document.documentElement.classList.toggle('light');
            localStorage.setItem('theme', document.documentElement.classList.contains('light') ? 'light' : 'dark');
        }}
        // Load saved theme preference
        if (localStorage.getItem('theme') === 'light') {{
            document.documentElement.classList.add('light');
            document.getElementById('themeToggle').checked = true;
        }}
    </script>
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
                // Check if left side is Neg - if so, preserve the Neg wrapper
                if let Expr::Neg(inner) = self.context.get(l).clone() {
                    // Traverse into the Neg and wrap result back in Neg
                    let new_inner =
                        self.reconstruct_global_expr(inner, remaining_path, replacement);
                    let new_neg = self.context.add(Expr::Neg(new_inner));
                    self.context.add(Expr::Add(new_neg, r))
                } else {
                    // Normal case
                    let new_l = self.reconstruct_global_expr(l, remaining_path, replacement);
                    self.context.add(Expr::Add(new_l, r))
                }
            }
            // Special case: Sub(a,b) may have been canonicalized to Add(a, Neg(b))
            // When PathStep::Right expects to modify the original "b", we need to
            // traverse into the Neg wrapper and reconstruct there.
            (Expr::Add(l, r), PathStep::Right) => {
                // Check if right side is Neg - if so, this might be a canonicalized Sub
                if let Expr::Neg(inner) = self.context.get(r).clone() {
                    // Traverse into the Neg and wrap result back in Neg
                    let new_inner =
                        self.reconstruct_global_expr(inner, remaining_path, replacement);
                    let new_neg = self.context.add(Expr::Neg(new_inner));
                    self.context.add(Expr::Add(l, new_neg))
                } else {
                    // Normal case - not a canonicalized Sub
                    let new_r = self.reconstruct_global_expr(r, remaining_path, replacement);
                    self.context.add(Expr::Add(l, new_r))
                }
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
                let mut new_args = args;
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

    /// Render timeline with enriched sub-steps (expandable details)
    fn render_timeline_filtered_enriched(
        &mut self,
        filtered_steps: &[&Step],
        enriched_steps: &[crate::didactic::EnrichedStep],
    ) -> String {
        let mut html = String::from("        <div class=\"timeline\">\n");

        // Build display hints for consistent sqrt notation (including final result)
        let display_hints = crate::display_context::build_display_context_with_result(
            self.context,
            self.original_expr,
            self.steps,
            self.simplified_result,
        );

        let mut step_number = 0;
        let mut last_global_after = self.original_expr; // Track final result across all steps
                                                        // Track if substeps have been shown (show only once on first visible step)
        let mut sub_steps_shown = false;

        // Track which steps to display
        let filtered_indices: std::collections::HashSet<_> =
            filtered_steps.iter().map(|s| *s as *const Step).collect();

        // Iterate over ALL steps to correctly update the global state
        for (step_idx, step) in self.steps.iter().enumerate() {
            // Use step.global_before/global_after if available (pre-computed with exponent simplification)
            // Otherwise fall back to recalculated state
            let global_before_expr = step.global_before.unwrap_or_else(|| {
                if step_idx == 0 {
                    self.original_expr
                } else {
                    // Reconstruct from previous step's global_after
                    self.steps
                        .get(step_idx - 1)
                        .and_then(|prev| prev.global_after)
                        .unwrap_or(self.original_expr)
                }
            });
            let global_after_expr = step.global_after.unwrap_or_else(|| {
                self.reconstruct_global_expr(global_before_expr, &step.path, step.after)
            });
            last_global_after = global_after_expr; // Always update for final result

            let step_ptr = step as *const Step;
            if !filtered_indices.contains(&step_ptr) {
                continue;
            }
            step_number += 1;

            // Generate global BEFORE with red highlight on the input (using display hints)
            let actual_target = self.find_expr_at_path(global_before_expr, &step.path);
            let mut before_config = HighlightConfig::new();
            before_config.add(actual_target, HighlightColor::Red);
            let global_before = cas_ast::LaTeXExprHighlightedWithHints {
                context: self.context,
                id: global_before_expr,
                highlights: &before_config,
                hints: &display_hints,
            }
            .to_latex();

            // Generate global AFTER with green highlight on the result (using display hints)
            let mut after_config = HighlightConfig::new();
            after_config.add(step.after, HighlightColor::Green);
            let global_after = cas_ast::LaTeXExprHighlightedWithHints {
                context: self.context,
                id: global_after_expr,
                highlights: &after_config,
                hints: &display_hints,
            }
            .to_latex();

            // Use LaTeXExprWithHints for proper LaTeX that respects sqrt() hints
            let local_before = cas_ast::LaTeXExprWithHints {
                context: self.context,
                id: step.before,
                hints: &display_hints,
            }
            .to_latex();
            let local_after = cas_ast::LaTeXExprWithHints {
                context: self.context,
                id: step.after,
                hints: &display_hints,
            }
            .to_latex();

            // Skip display no-op steps where before and after render identically
            if local_before == local_after {
                step_number -= 1; // Undo the increment since we're skipping
                continue;
            }

            let local_change_latex = format!("{} \\rightarrow {}", local_before, local_after);

            // Get enriched sub-steps for this step
            // Detect enrichment type FIRST
            let sub_steps_html = if let Some(enriched) = enriched_steps.get(step_idx) {
                if !enriched.sub_steps.is_empty() {
                    // Detect type from sub-step descriptions
                    let has_fraction_sum = enriched.sub_steps.iter().any(|s| {
                        s.description.contains("common denominator")
                            || s.description.contains("Sum the fractions")
                    });
                    let has_factorization = enriched.sub_steps.iter().any(|s| {
                        s.description.contains("Cancel common factor")
                            || s.description.contains("Factor")
                    });
                    let has_nested_fraction = enriched.sub_steps.iter().any(|s| {
                        s.description.contains("Invertir") || s.description.contains("denominador")
                    });

                    // Per-step enrichments (nested fractions, factorization): always show
                    // Global enrichments (fraction sums): show only once
                    let should_show = if has_nested_fraction || has_factorization {
                        true // Per-step: always show for each relevant step
                    } else {
                        !sub_steps_shown // Global (fraction sums or default): only show once
                    };

                    if should_show {
                        // Mark as shown for global enrichments only
                        if has_fraction_sum && !has_nested_fraction && !has_factorization {
                            sub_steps_shown = true;
                        }

                        let header = if has_nested_fraction {
                            "Simplificación de fracción compleja"
                        } else if has_fraction_sum {
                            "Suma de fracciones"
                        } else if has_factorization {
                            "Factorización de polinomios"
                        } else {
                            "Pasos intermedios"
                        };

                        let mut details_html = format!(
                            r#"<details class="substeps-details">
                            <summary>{}</summary>
                            <div class="substeps-content">"#,
                            header
                        );
                        for sub in &enriched.sub_steps {
                            details_html.push_str(&format!(
                                r#"<div class="substep">
                                    <span class="substep-desc">{}</span>"#,
                                html_escape(&sub.description)
                            ));
                            if !sub.before_latex.is_empty() {
                                details_html.push_str(&format!(
                                    r#"<div class="substep-math">\[{} \rightarrow {}\]</div>"#,
                                    sub.before_latex, sub.after_latex
                                ));
                            }
                            details_html.push_str("</div>");
                        }
                        details_html.push_str("</div></details>");
                        details_html
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            // Build domain warning HTML from assumption_events
            let domain_html = if !step.assumption_events.is_empty() {
                let messages: Vec<String> = step
                    .assumption_events
                    .iter()
                    .map(|e| html_escape(&e.message))
                    .collect();
                format!(
                    r#"                    <div class="domain-warning">Domain: {}</div>
"#,
                    messages.join(", ")
                )
            } else {
                String::new()
            };

            // Build requires HTML from required_conditions (implicit domain constraints)
            let requires_html = if !step.required_conditions.is_empty() {
                let messages: Vec<String> = step
                    .required_conditions
                    .iter()
                    .map(|c| html_escape(&c.display(self.context)))
                    .collect();
                format!(
                    r#"                    <div class="domain-requires">Requires: {}</div>
"#,
                    messages.join(", ")
                )
            } else {
                String::new()
            };

            html.push_str(&format!(
                r#"            <div class="step">
                <div class="step-number">{}</div>
                <div class="step-content">
                    <h3>{}</h3>
                    <div class="math-expr before">
                        \(\textbf{{Before:}}\)
                        \[{}\]
                    </div>
                    {}
                    <div class="rule-description">
                        <div class="rule-name">\(\text{{{}}}\)</div>
                        <div class="local-change">
                            \[{}\]
                        </div>
                    </div>
                    <div class="math-expr after">
                        \(\textbf{{After:}}\)
                        \[{}\]
                    </div>
{}{}                </div>
            </div>
"#,
                step_number,
                html_escape(&step.rule_name),
                global_before,
                sub_steps_html,
                step.description,
                local_change_latex,
                global_after,
                requires_html,
                domain_html
            ));
        }

        // Add final result with display hints for consistent root notation
        // Use simplified_result if available (passed from simplifier), otherwise use last_global_after
        let final_result_expr = self.simplified_result.unwrap_or(last_global_after);
        let final_expr = cas_ast::LaTeXExprWithHints {
            context: self.context,
            id: final_result_expr,
            hints: &display_hints,
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
pub fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Prepare string for LaTeX rendering in MathJax
pub fn latex_escape(s: &str) -> String {
    // For MathJax, we mostly just need the string as-is
    // But escape backslashes that aren't part of LaTeX commands
    s.to_string()
}

// ============================================================================
// SolveTimelineHtml - Timeline for equation solving steps
// ============================================================================

use crate::solver::SolveStep;
use cas_ast::{Equation, SolutionSet};

/// Timeline HTML generator for equation solving steps
pub struct SolveTimelineHtml<'a> {
    context: &'a mut Context,
    steps: &'a [SolveStep],
    original_eq: &'a Equation,
    solution_set: &'a SolutionSet,
    var: String,
    title: String,
}

impl<'a> SolveTimelineHtml<'a> {
    pub fn new(
        context: &'a mut Context,
        steps: &'a [SolveStep],
        original_eq: &'a Equation,
        solution_set: &'a SolutionSet,
        var: &str,
    ) -> Self {
        let title = format!(
            "{} {} {}",
            DisplayExpr {
                context,
                id: original_eq.lhs
            },
            original_eq.op,
            DisplayExpr {
                context,
                id: original_eq.rhs
            }
        );
        Self {
            context,
            steps,
            original_eq,
            solution_set,
            var: var.to_string(),
            title,
        }
    }

    /// Generate complete HTML document for solve steps
    pub fn to_html(&mut self) -> String {
        let mut html = self.html_header_solve();
        html.push_str(&self.render_solve_timeline());
        html.push_str(Self::html_footer_solve());
        html
    }

    fn html_header_solve(&self) -> String {
        let escaped_title = html_escape(&self.title);
        let original_latex = format!(
            "{} {} {}",
            LaTeXExpr {
                context: self.context,
                id: self.original_eq.lhs
            }
            .to_latex(),
            self.relop_to_latex(&self.original_eq.op),
            LaTeXExpr {
                context: self.context,
                id: self.original_eq.rhs
            }
            .to_latex()
        );

        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Solve Steps: {}</title>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
    * {{
        box-sizing: border-box;
    }}
    /* Theme variables */
    :root {{
        --bg-gradient-start: #1a1a2e;
        --bg-gradient-end: #16213e;
        --container-bg: rgba(30, 40, 60, 0.95);
        --container-shadow: rgba(0, 0, 0, 0.3);
        --title-color: #64b5f6;
        --subtitle-color: #90caf9;
        --original-bg-start: #1565c0;
        --original-bg-end: #0d47a1;
        --original-shadow: rgba(21, 101, 192, 0.4);
        --timeline-line-start: #64b5f6;
        --timeline-line-end: #4caf50;
        --step-bg: rgba(40, 50, 70, 0.8);
        --step-border: #64b5f6;
        --step-hover-shadow: rgba(100, 181, 246, 0.3);
        --step-dot-bg: #64b5f6;
        --step-dot-border: #1a1a2e;
        --step-number-color: #64b5f6;
        --description-color: #b0bec5;
        --equation-bg: rgba(30, 40, 55, 0.9);
        --final-bg-start: #2e7d32;
        --final-bg-end: #1b5e20;
        --final-shadow: rgba(76, 175, 80, 0.3);
        --footer-color: white;
        --text-color: #e0e0e0;
    }}
    :root.light {{
        --bg-gradient-start: #667eea;
        --bg-gradient-end: #764ba2;
        --container-bg: white;
        --container-shadow: rgba(0, 0, 0, 0.2);
        --title-color: #333;
        --subtitle-color: #666;
        --original-bg-start: #f0f4ff;
        --original-bg-end: #f0f4ff;
        --original-shadow: rgba(102, 126, 234, 0.2);
        --timeline-line-start: #667eea;
        --timeline-line-end: #764ba2;
        --step-bg: white;
        --step-border: #667eea;
        --step-hover-shadow: rgba(102, 126, 234, 0.3);
        --step-dot-bg: #667eea;
        --step-dot-border: white;
        --step-number-color: #667eea;
        --description-color: #666;
        --equation-bg: #fafafa;
        --final-bg-start: #4caf50;
        --final-bg-end: #45a049;
        --final-shadow: rgba(76, 175, 80, 0.3);
        --footer-color: white;
        --text-color: #333;
    }}
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px 15px;
        background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
        min-height: 100vh;
        color: var(--text-color);
        transition: background 0.3s ease;
    }}
    .container {{
        background: var(--container-bg);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px var(--container-shadow);
        transition: background 0.3s ease;
    }}
    /* Theme toggle switch */
    .theme-toggle {{
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 8px;
        background: var(--container-bg);
        padding: 8px 12px;
        border-radius: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }}
    .theme-toggle span {{
        font-size: 1.2em;
    }}
    .toggle-switch {{
        position: relative;
        width: 50px;
        height: 26px;
    }}
    .toggle-switch input {{
        opacity: 0;
        width: 0;
        height: 0;
    }}
    .toggle-slider {{
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: #333;
        transition: 0.3s;
        border-radius: 26px;
    }}
    .toggle-slider:before {{
        position: absolute;
        content: "";
        height: 20px;
        width: 20px;
        left: 3px;
        bottom: 3px;
        background-color: white;
        transition: 0.3s;
        border-radius: 50%;
    }}
    input:checked + .toggle-slider {{
        background-color: #64b5f6;
    }}
    input:checked + .toggle-slider:before {{
        transform: translateX(24px);
    }}
    h1 {{
        color: var(--title-color);
        text-align: center;
        margin-bottom: 10px;
        font-size: 1.8em;
        transition: color 0.3s ease;
    }}
    .subtitle {{
        text-align: center;
        color: var(--subtitle-color);
        margin-bottom: 25px;
        transition: color 0.3s ease;
    }}
    .original {{
        background: linear-gradient(135deg, var(--original-bg-start), var(--original-bg-end));
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 4px 15px var(--original-shadow);
        transition: background 0.3s ease;
    }}
    .timeline {{
        position: relative;
        padding-left: 30px;
    }}
    .timeline::before {{
        content: '';
        position: absolute;
        left: 10px;
        top: 0;
        bottom: 0;
        width: 3px;
        background: linear-gradient(to bottom, var(--timeline-line-start), var(--timeline-line-end));
        transition: background 0.3s ease;
    }}
    .step {{
        background: var(--step-bg);
        border-radius: 10px;
        padding: 15px 20px;
        margin-bottom: 20px;
        position: relative;
        border-left: 4px solid var(--step-border);
        transition: transform 0.2s, box-shadow 0.2s, background 0.3s ease;
    }}
    .step:hover {{
        transform: translateX(5px);
        box-shadow: 0 4px 20px var(--step-hover-shadow);
    }}
    .step::before {{
        content: '';
        position: absolute;
        left: -23px;
        top: 20px;
        width: 12px;
        height: 12px;
        background: var(--step-dot-bg);
        border-radius: 50%;
        border: 3px solid var(--step-dot-border);
        transition: background 0.3s ease;
    }}
    .step-number {{
        color: var(--step-number-color);
        font-weight: bold;
        font-size: 0.9em;
        margin-bottom: 5px;
        transition: color 0.3s ease;
    }}
    .description {{
        color: var(--description-color);
        font-size: 1em;
        margin-bottom: 10px;
        font-style: italic;
        transition: color 0.3s ease;
    }}
    .equation {{
        background: var(--equation-bg);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2em;
        transition: background 0.3s ease;
    }}
    .final-result {{
        background: linear-gradient(135deg, var(--final-bg-start), var(--final-bg-end));
        padding: 20px;
        text-align: center;
        color: white;
        border-radius: 10px;
        margin-top: 30px;
        font-size: 1.2em;
        box-shadow: 0 4px 12px var(--final-shadow);
        transition: background 0.3s ease;
    }}
    footer {{
        text-align: center;
        margin-top: 30px;
        color: var(--footer-color);
        font-size: 0.9em;
        transition: color 0.3s ease;
    }}
</style>
</head>
<body>
<div class="theme-toggle">
    <span>🌙</span>
    <label class="toggle-switch">
        <input type="checkbox" id="themeToggle" onchange="toggleTheme()">
        <span class="toggle-slider"></span>
    </label>
    <span>☀️</span>
</div>
<script>
    function toggleTheme() {{
        document.documentElement.classList.toggle('light');
        localStorage.setItem('theme', document.documentElement.classList.contains('light') ? 'light' : 'dark');
    }}
    // Load saved theme preference
    if (localStorage.getItem('theme') === 'light') {{
        document.documentElement.classList.add('light');
        document.getElementById('themeToggle').checked = true;
    }}
</script>
<div class="container">
    <h1>Equation Solving Steps</h1>
    <p class="subtitle">Solving for <strong>{}</strong></p>
    <div class="original">
        \(\textbf{{Original Equation:}}\)
        \[{}\]
    </div>
"#,
            escaped_title,
            html_escape(&self.var),
            original_latex
        )
    }

    fn render_solve_timeline(&mut self) -> String {
        let mut html = String::from("        <div class=\"timeline\">\n");

        let mut _current_eq = self.original_eq.clone();

        for (i, step) in self.steps.iter().enumerate() {
            let step_number = i + 1;

            // Generate LaTeX for the equation after this step
            let eq_latex = format!(
                "{} {} {}",
                LaTeXExpr {
                    context: self.context,
                    id: step.equation_after.lhs
                }
                .to_latex(),
                self.relop_to_latex(&step.equation_after.op),
                LaTeXExpr {
                    context: self.context,
                    id: step.equation_after.rhs
                }
                .to_latex()
            );

            html.push_str(&format!(
                r#"        <div class="step">
            <div class="step-number">Step {}</div>
            <div class="description">{}</div>
            <div class="equation">
                \[{}\]
            </div>
        </div>
"#,
                step_number,
                html_escape(&step.description),
                eq_latex
            ));

            _current_eq = step.equation_after.clone();
        }

        // Add final result showing the SOLUTION SET, not the last equation
        let solution_latex = self.solution_set_to_latex();

        html.push_str(&format!(
            r#"        </div>
        <div class="final-result">
            \(\textbf{{Solution: }} {} = \)
            \[{}\]
        </div>
    </div>
"#,
            html_escape(&self.var),
            solution_latex
        ));
        html
    }

    /// Convert SolutionSet to LaTeX representation
    fn solution_set_to_latex(&self) -> String {
        use cas_ast::SolutionSet;
        match self.solution_set {
            SolutionSet::Empty => r"\emptyset".to_string(),
            SolutionSet::AllReals => r"\mathbb{R}".to_string(),
            SolutionSet::Discrete(exprs) => {
                let elements: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        LaTeXExpr {
                            context: self.context,
                            id: *e,
                        }
                        .to_latex()
                    })
                    .collect();
                format!(r"\left\{{ {} \right\}}", elements.join(", "))
            }
            SolutionSet::Continuous(interval) => self.interval_to_latex(interval),
            SolutionSet::Union(intervals) => {
                let parts: Vec<String> = intervals
                    .iter()
                    .map(|i| self.interval_to_latex(i))
                    .collect();
                parts.join(r" \cup ")
            }
            SolutionSet::Residual(expr) => {
                // Show the residual expression as-is
                LaTeXExpr {
                    context: self.context,
                    id: *expr,
                }
                .to_latex()
            }
            SolutionSet::Conditional(cases) => {
                // V2.0 Phase 2C: Pretty-print conditional solutions as piecewise LaTeX
                // V2.1: Use "otherwise" without "if" prefix for natural reading
                let case_strs: Vec<String> = cases
                    .iter()
                    .map(|case| {
                        let sol_latex = self.solution_set_inner_to_latex(&case.then.solutions);
                        if case.when.is_otherwise() {
                            format!("{} & \\text{{otherwise}}", sol_latex)
                        } else {
                            let cond_latex = case.when.latex_display_with_context(self.context);
                            format!("{} & \\text{{if }} {}", sol_latex, cond_latex)
                        }
                    })
                    .collect();
                format!(r"\begin{{cases}} {} \end{{cases}}", case_strs.join(r" \\ "))
            }
        }
    }

    /// V2.0 Phase 2C: Render inner solution set to LaTeX (for Conditional cases)
    fn solution_set_inner_to_latex(&self, solution_set: &SolutionSet) -> String {
        match solution_set {
            SolutionSet::Empty => r"\emptyset".to_string(),
            SolutionSet::AllReals => r"\mathbb{R}".to_string(),
            SolutionSet::Discrete(exprs) => {
                let elements: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        LaTeXExpr {
                            context: self.context,
                            id: *e,
                        }
                        .to_latex()
                    })
                    .collect();
                format!(r"\left\{{ {} \right\}}", elements.join(", "))
            }
            SolutionSet::Continuous(interval) => self.interval_to_latex(interval),
            SolutionSet::Union(intervals) => {
                let parts: Vec<String> = intervals
                    .iter()
                    .map(|i| self.interval_to_latex(i))
                    .collect();
                parts.join(r" \cup ")
            }
            SolutionSet::Residual(expr) => LaTeXExpr {
                context: self.context,
                id: *expr,
            }
            .to_latex(),
            SolutionSet::Conditional(_) => r"\text{(nested conditional)}".to_string(),
        }
    }

    /// Convert Interval to LaTeX representation
    fn interval_to_latex(&self, interval: &cas_ast::Interval) -> String {
        use cas_ast::BoundType;
        let left = match interval.min_type {
            BoundType::Open => "(",
            BoundType::Closed => "[",
        };
        let right = match interval.max_type {
            BoundType::Open => ")",
            BoundType::Closed => "]",
        };
        let min_latex = LaTeXExpr {
            context: self.context,
            id: interval.min,
        }
        .to_latex();
        let max_latex = LaTeXExpr {
            context: self.context,
            id: interval.max,
        }
        .to_latex();
        format!(r"{}{}, {}{}", left, min_latex, max_latex, right)
    }

    fn relop_to_latex(&self, op: &cas_ast::RelOp) -> &'static str {
        use cas_ast::RelOp;
        match op {
            RelOp::Eq => "=",
            RelOp::Neq => "\\neq",
            RelOp::Lt => "<",
            RelOp::Gt => ">",
            RelOp::Leq => "\\leq",
            RelOp::Geq => "\\geq",
        }
    }

    fn html_footer_solve() -> &'static str {
        r#"    <footer>
        Generated by Rust CAS Engine - Equation Solver
    </footer>
</body>
</html>"#
    }
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
