use crate::poly_store::try_render_poly_result_latex;
use crate::step::{pathsteps_to_expr_path, PathStep, Step};
use cas_ast::{
    Context, DisplayExpr, Expr, ExprId, ExprPath, HighlightColor, HighlightConfig, LaTeXExpr,
    LaTeXExprHighlighted, PathHighlightConfig, PathHighlightedLatexRenderer,
};
use num_traits::Signed;

/// Convert expression to LaTeX, intercepting poly_result for direct rendering.
/// If the expression is a poly_result, renders the polynomial directly as LaTeX.
/// Otherwise, falls back to standard LaTeXExpr rendering.
#[allow(dead_code)]
fn expr_to_latex_or_poly(ctx: &Context, id: ExprId) -> String {
    if let Some(latex) = try_render_poly_result_latex(ctx, id) {
        return latex;
    }
    LaTeXExpr { context: ctx, id }.to_latex()
}

/// Convert PathStep to u8 for ExprPath (V2.9.17)
#[inline]
fn pathstep_to_u8(ps: &PathStep) -> u8 {
    ps.to_child_index()
}

/// Find path from root to target expression (V2.9.17)
/// Returns empty Vec if target not found or target == root
fn find_path_to_expr(ctx: &Context, root: ExprId, target: ExprId) -> Vec<PathStep> {
    if root == target {
        return vec![];
    }

    // DFS to find target within root
    fn dfs(ctx: &Context, current: ExprId, target: ExprId, path: &mut Vec<PathStep>) -> bool {
        if current == target {
            return true;
        }

        match ctx.get(current) {
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                path.push(PathStep::Left);
                if dfs(ctx, *l, target, path) {
                    return true;
                }
                path.pop();

                path.push(PathStep::Right);
                if dfs(ctx, *r, target, path) {
                    return true;
                }
                path.pop();
            }
            Expr::Pow(base, exp) => {
                path.push(PathStep::Base);
                if dfs(ctx, *base, target, path) {
                    return true;
                }
                path.pop();

                path.push(PathStep::Exponent);
                if dfs(ctx, *exp, target, path) {
                    return true;
                }
                path.pop();
            }
            Expr::Neg(inner) => {
                path.push(PathStep::Inner);
                if dfs(ctx, *inner, target, path) {
                    return true;
                }
                path.pop();
            }
            Expr::Function(_, args) => {
                for (i, arg) in args.iter().enumerate() {
                    path.push(PathStep::Arg(i));
                    if dfs(ctx, *arg, target, path) {
                        return true;
                    }
                    path.pop();
                }
            }
            _ => {}
        }
        false
    }

    let mut path = Vec::new();
    if dfs(ctx, root, target, &mut path) {
        path
    } else {
        vec![]
    }
}

/// Extract terms from an Add/Sub chain for multi-term highlighting (V2.9.17)
/// Returns individual ExprIds that make up the chain.
/// Also unwraps Neg wrappers since they may be dynamically created and not exist
/// in the original expression tree.
fn extract_add_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    fn collect_terms(ctx: &Context, id: ExprId, terms: &mut Vec<ExprId>) {
        match ctx.get(id) {
            Expr::Add(l, r) => {
                collect_terms(ctx, *l, terms);
                collect_terms(ctx, *r, terms);
            }
            Expr::Neg(inner) => {
                // Unwrap Neg to find the underlying term (which may exist in original tree)
                // Add both: the Neg itself (if it exists) and the inner term
                terms.push(id); // The Neg wrapper
                terms.push(*inner); // The inner term (more likely to exist in original)
            }
            _ => terms.push(id),
        }
    }
    collect_terms(ctx, expr, &mut terms);
    terms
}

/// Diff-based fallback: find path to target expression within a tree (V2.9.18)
/// When direct path lookup fails, this performs a depth-first search to find
/// where the target expression appears in the tree.
/// Returns the path if found, or None if not found.
fn diff_find_path_to_expr(ctx: &Context, root: ExprId, target: ExprId) -> Option<ExprPath> {
    // First, try direct ExprId match
    if root == target {
        return Some(vec![]);
    }

    // Recursively search children
    fn search(ctx: &Context, current: ExprId, target: ExprId, path: &mut ExprPath) -> bool {
        if current == target {
            return true;
        }

        match ctx.get(current) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                // Try left branch
                path.push(0); // Left = 0
                if search(ctx, *l, target, path) {
                    return true;
                }
                path.pop();

                // Try right branch
                path.push(1); // Right = 1
                if search(ctx, *r, target, path) {
                    return true;
                }
                path.pop();
            }
            Expr::Neg(inner) => {
                path.push(0);
                if search(ctx, *inner, target, path) {
                    return true;
                }
                path.pop();
            }
            Expr::Function(_, args) => {
                for (i, arg) in args.iter().enumerate() {
                    path.push(i as u8);
                    if search(ctx, *arg, target, path) {
                        return true;
                    }
                    path.pop();
                }
            }
            _ => {}
        }
        false
    }

    let mut path = vec![];
    if search(ctx, root, target, &mut path) {
        Some(path)
    } else {
        None
    }
}

/// Find ALL paths to a target expression within a tree (V2.9.19)
/// Unlike diff_find_path_to_expr which returns only the first match,
/// this finds every occurrence (useful for x+x where both x share same ExprId in DAG)
fn diff_find_all_paths_to_expr(ctx: &Context, root: ExprId, target: ExprId) -> Vec<ExprPath> {
    let mut results = Vec::new();

    fn search(
        ctx: &Context,
        current: ExprId,
        target: ExprId,
        path: &mut ExprPath,
        results: &mut Vec<ExprPath>,
    ) {
        if current == target {
            results.push(path.clone());
            // Continue searching - there may be more occurrences below or in siblings
        }

        match ctx.get(current) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                // Search left branch
                path.push(0);
                search(ctx, *l, target, path, results);
                path.pop();

                // Search right branch
                path.push(1);
                search(ctx, *r, target, path, results);
                path.pop();
            }
            Expr::Neg(inner) => {
                path.push(0);
                search(ctx, *inner, target, path, results);
                path.pop();
            }
            Expr::Function(_, args) => {
                for (i, arg) in args.iter().enumerate() {
                    path.push(i as u8);
                    search(ctx, *arg, target, path, results);
                    path.pop();
                }
            }
            _ => {}
        }
    }

    let mut path = vec![];
    search(ctx, root, target, &mut path, &mut results);
    results
}

/// Find paths to a target expression by structural equivalence (V2.9.24)
/// Unlike diff_find_all_paths_to_expr which matches by ExprId,
/// this uses compare_expr for structural comparison.
/// Essential for dynamically constructed before_local expressions
/// where terms may have different ExprIds than the original tree.
fn diff_find_paths_by_structure(ctx: &Context, root: ExprId, target: ExprId) -> Vec<ExprPath> {
    let mut results = Vec::new();

    fn search(
        ctx: &Context,
        current: ExprId,
        target: ExprId,
        path: &mut ExprPath,
        results: &mut Vec<ExprPath>,
    ) {
        // Check structural equivalence using compare_expr
        if crate::ordering::compare_expr(ctx, current, target) == std::cmp::Ordering::Equal {
            results.push(path.clone());
            // Don't recurse into children if we matched the whole subtree
            return;
        }

        match ctx.get(current) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                // Search left branch
                path.push(0);
                search(ctx, *l, target, path, results);
                path.pop();

                // Search right branch
                path.push(1);
                search(ctx, *r, target, path, results);
                path.pop();
            }
            Expr::Neg(inner) => {
                path.push(0);
                search(ctx, *inner, target, path, results);
                path.pop();
            }
            Expr::Function(_, args) => {
                for (i, arg) in args.iter().enumerate() {
                    path.push(i as u8);
                    search(ctx, *arg, target, path, results);
                    path.pop();
                }
            }
            _ => {}
        }
    }

    let mut path = vec![];
    search(ctx, root, target, &mut path, &mut results);
    results
}

/// Navigate to the subexpression at a given path within an expression tree
fn navigate_to_subexpr(ctx: &Context, mut current: ExprId, path: &ExprPath) -> ExprId {
    for &step in path {
        match ctx.get(current) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                current = if step == 0 { *l } else { *r };
            }
            Expr::Neg(inner) => {
                current = *inner;
            }
            Expr::Function(_, args) => {
                if let Some(&arg) = args.get(step as usize) {
                    current = arg;
                }
            }
            _ => break,
        }
    }
    current
}

/// Clean identity patterns from LaTeX strings (removes redundant ·1 patterns)
/// This is the LaTeX equivalent of clean_display_string for CLI output.
/// Patterns like "\cdot 1" and "1 \cdot x" are cleaned up for better display.
fn clean_latex_identities(latex: &str) -> String {
    use regex::Regex;

    let mut result = latex.to_string();
    let mut changed = true;

    // Pre-compile regex outside the loop (clippy: regex_creation_in_loops)
    let re_mult_unit_frac = Regex::new(r"(\d+)\s*\\cdot\s*\\frac\{1\}\{([^}]+)\}").unwrap();

    // Iterate until no more changes (handles nested patterns)
    while changed {
        let before = result.clone();

        // "\cdot 1}" at end of group → "}" (remove trailing ·1)
        result = result.replace("\\cdot 1}", "}");

        // "\cdot 1$" at end of math (inline) → "$"
        result = result.replace("\\cdot 1$", "$");

        // "\cdot 1 " (with space after) → " "
        result = result.replace("\\cdot 1 ", " ");

        // "\cdot 1+" → "+"
        result = result.replace("\\cdot 1+", "+");

        // "\cdot 1-" → "-"
        result = result.replace("\\cdot 1-", "-");

        // "1 \cdot " at start or after operators → ""
        result = result.replace("{1 \\cdot ", "{");

        // Handle pattern at start of string
        if result.starts_with("1 \\cdot ") {
            result = result[8..].to_string();
        }

        // "\cdot 1\" (before another LaTeX command) → "\"
        result = result.replace("\\cdot 1\\", "\\");

        // Handle "\frac{1}{1}" → "1"
        result = result.replace("\\frac{1}{1}", "1");

        // Standalone "\cdot 1" at the very end
        if result.ends_with("\\cdot 1") {
            result = result[..result.len() - 7].to_string();
        }

        // KEY FIX: Convert "n \cdot \frac{1}{expr}" → "\frac{n}{expr}"
        // This handles cases like "2 \cdot \frac{1}{x}" → "\frac{2}{x}"
        result = re_mult_unit_frac
            .replace_all(&result, r"\frac{$1}{$2}")
            .to_string();

        changed = before != result;
    }

    result
}

/// Timeline HTML generator - exports simplification steps to interactive HTML
pub struct TimelineHtml<'a> {
    context: &'a mut Context,
    steps: &'a [Step],
    original_expr: ExprId,
    simplified_result: Option<ExprId>, // Optional: the final simplified result
    title: String,
    verbosity_level: VerbosityLevel,
    /// V2.12.13: Global requires inferred from input expression.
    /// Shown at the end of the timeline, after final result.
    global_requires: Vec<crate::implicit_domain::ImplicitCondition>,
    /// V2.14.40: Style preferences derived from input string for consistent root rendering
    style_prefs: cas_ast::StylePreferences,
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
        Self::new_with_result_and_style(
            context,
            steps,
            original_expr,
            simplified_result,
            verbosity,
            None,
        )
    }

    /// Create a new TimelineHtml with style preferences derived from input string
    /// V2.14.40: Enables consistent root rendering (exponential vs radical)
    pub fn new_with_result_and_style(
        context: &'a mut Context,
        steps: &'a [Step],
        original_expr: ExprId,
        simplified_result: Option<ExprId>,
        verbosity: VerbosityLevel,
        input_string: Option<&str>,
    ) -> Self {
        use crate::implicit_domain::infer_implicit_domain;
        use crate::semantics::ValueDomain;

        // V2.14.40: Compute style preferences from input string
        let signals = input_string.map(cas_ast::ParseStyleSignals::from_input_string);
        let style_prefs = cas_ast::StylePreferences::from_expression_with_signals(
            context,
            original_expr,
            signals.as_ref(),
        );

        // V2.14.40: Generate title using styled renderer for consistent root rendering
        let empty_config = PathHighlightConfig::new();
        let title = PathHighlightedLatexRenderer {
            context,
            id: original_expr,
            path_highlights: &empty_config,
            hints: None,
            style_prefs: Some(&style_prefs),
        }
        .to_latex();

        // V2.12.13: Infer global requires from input expression
        // This ensures timeline shows the same requires as REPL
        let input_domain = infer_implicit_domain(context, original_expr, ValueDomain::RealOnly);
        let global_requires: Vec<_> = input_domain.conditions().iter().cloned().collect();

        Self {
            context,
            steps,
            original_expr,
            simplified_result,
            title,
            verbosity_level: verbosity,
            global_requires,
            style_prefs,
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
        use regex::Regex;
        let mut result = latex.to_string();

        // Fix "+ -" → "-" in all contexts
        result = result.replace("+ -\\", "- \\"); // Before LaTeX commands
        result = result.replace("+ -(", "- ("); // Before parentheses

        // Fix "- -" → "+" (double negative)
        result = result.replace("- -\\", "+ \\"); // Before LaTeX commands
        result = result.replace("- -(", "+ ("); // Before parentheses

        // Fix "+ -" before digits or letters (e.g., "+ -x" → "- x")
        let re_plus_minus = Regex::new(r"\+ -([0-9a-zA-Z])").unwrap();
        result = re_plus_minus.replace_all(&result, "- $1").to_string();

        // Fix "- -" before digits or letters (e.g., "- -x" → "+ x")
        let re_minus_minus = Regex::new(r"- -([0-9a-zA-Z])").unwrap();
        result = re_minus_minus.replace_all(&result, "+ $1").to_string();

        // Fix "+ {color command}{-" patterns (highlighted negatives)
        // e.g., "+ {\color{red}{-..." → "- {\color{red}{"
        let re_plus_color_minus = Regex::new(r"\+ (\{\\color\{[^}]+\}\{)-").unwrap();
        result = re_plus_color_minus.replace_all(&result, "- $1").to_string();

        // Fix "- {color command}{-" patterns (double negative with highlight)
        let re_minus_color_minus = Regex::new(r"- (\{\\color\{[^}]+\}\{)-").unwrap();
        result = re_minus_color_minus
            .replace_all(&result, "+ $1")
            .to_string();

        // Fix "+ -{" → "- {" (when minus precedes a brace group)
        result = result.replace("+ -{", "- {");

        // Fix "- -{" → "+ {" (double negative before brace)
        result = result.replace("- -{", "+ {");

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

                let fn_name = self.context.sym_name(*name);
                match fn_name {
                    "sqrt" if highlighted_args.len() == 1 => {
                        format!("\\sqrt{{{}}}", highlighted_args[0])
                    }
                    "sqrt" if highlighted_args.len() == 2 => {
                        format!("\\sqrt[{}]{{{}}}", highlighted_args[1], highlighted_args[0])
                    }
                    "sin" | "cos" | "tan" | "cot" | "sec" | "csc" => {
                        format!("\\{}({})", fn_name, highlighted_args[0])
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

        // Clean up identity patterns like "\cdot 1" for better display
        clean_latex_identities(&html)
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
            max-width: 95%;
            margin: 0 auto;
            padding: 20px 10px;
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
            overflow-x: auto;
            max-width: 100%;
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
        .poly-badge {{
            background: rgba(255,255,255,0.25);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75em;
            margin-left: 10px;
            font-weight: normal;
        }}
        .poly-output {{
            text-align: left;
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.65em;
            max-height: 400px;
            overflow: auto;
            white-space: pre-wrap;
            word-break: break-all;
            line-height: 1.4;
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
        .global-requires {{
            margin-top: 15px;
            padding: 12px 16px;
            background: rgba(33, 150, 243, 0.1);
            border: 2px solid rgba(33, 150, 243, 0.5);
            border-radius: 8px;
            color: #90caf9;
            font-size: 1em;
        }}
        .global-requires strong {{
            color: #64b5f6;
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

            // Generate global BEFORE with red highlight on the transformed subtree
            // V2.9.16: Using PathHighlightedLatexRenderer to highlight by path, not ExprId
            // This ensures only the specific occurrence is highlighted, not all identical values
            //
            // V2.9.17: When before_local differs from step.before, try to extend path to focus area
            // If path cannot be found (before_local is dynamically constructed), use ExprId-based highlighting
            //
            // V2.9.25: When before_local is an Add node (representing multiple matched terms),
            // always use multi-term highlighting. Single-path highlighting would only highlight
            // one subtree when the matched terms may come from different parts of the expression.
            let (global_before, global_after) = if step.before_local.is_some()
                && step.before_local != Some(step.before)
            {
                let before_local = step.before_local.unwrap();

                // V2.9.25: Check if before_local is an Add node. If so, use multi-term highlighting
                // to ensure all terms are highlighted, not just the subtree that happens to match.
                let before_local_is_add = matches!(self.context.get(before_local), Expr::Add(_, _));

                // Try to find path from step.before to before_local
                let focus_path = if !before_local_is_add {
                    find_path_to_expr(self.context, step.before, before_local)
                } else {
                    // Skip single-path search for Add nodes - will use multi-term highlighting
                    Vec::new()
                };

                if !focus_path.is_empty() {
                    // Path found - extend step.path and use path-based highlighting
                    // This branch is only used for non-Add before_local nodes
                    let mut extended = pathsteps_to_expr_path(&step.path);
                    for ps in &focus_path {
                        extended.push(pathstep_to_u8(ps));
                    }
                    let mut before_config = PathHighlightConfig::new();
                    before_config.add(extended.clone(), HighlightColor::Red);
                    let before = PathHighlightedLatexRenderer {
                        context: self.context,
                        id: global_before_expr,
                        path_highlights: &before_config,
                        hints: Some(&display_hints),
                        style_prefs: Some(&self.style_prefs),
                    }
                    .to_latex();

                    let mut after_config = PathHighlightConfig::new();
                    after_config.add(extended, HighlightColor::Green);
                    let after = PathHighlightedLatexRenderer {
                        context: self.context,
                        id: global_after_expr,
                        path_highlights: &after_config,
                        hints: Some(&display_hints),
                        style_prefs: Some(&self.style_prefs),
                    }
                    .to_latex();

                    (before, after)
                } else {
                    // Path not found (before_local is dynamically constructed)
                    // V2.9.19: Use multi-path highlighting with paths to individual terms
                    // This fixes the regression where ExprId-based highlighting would mark
                    // all identical values (e.g., all 'x' symbols) instead of just those
                    // within the focus area.
                    let focus_before = step.before_local.unwrap();
                    let focus_after = step.after_local.unwrap_or(step.after);

                    // BEFORE: Extract terms from focus_before and find paths to each within
                    // the subexpression at step.path (NOT the entire global_before_expr).
                    // This handles dynamically constructed expressions like Add(x, x) or Sub(frac1, frac2)
                    let focus_terms = extract_add_terms(self.context, focus_before);
                    let step_path_prefix = pathsteps_to_expr_path(&step.path);

                    // Navigate to the subexpression at step.path
                    let subexpr_at_path =
                        navigate_to_subexpr(self.context, global_before_expr, &step_path_prefix);

                    // V2.14.32: Find the path to before_local within subexpr_at_path
                    // This limits the search scope to only the part that's actually being transformed.
                    // For example, when transforming a numerator, we should not highlight
                    // occurrences in the denominator that happen to share an ExprId.
                    let before_local_path =
                        diff_find_path_to_expr(self.context, subexpr_at_path, focus_before);

                    // Determine the actual scope: either before_local subtree or full subexpr_at_path
                    let (search_scope, scope_path_prefix) =
                        if let Some(path_to_local) = &before_local_path {
                            // before_local exists in the tree - limit search to that subtree
                            let local_scope =
                                navigate_to_subexpr(self.context, subexpr_at_path, path_to_local);
                            let mut full_prefix = step_path_prefix.clone();
                            full_prefix.extend(path_to_local.clone());
                            (local_scope, full_prefix)
                        } else {
                            // before_local is dynamically constructed - use full subexpr_at_path
                            (subexpr_at_path, step_path_prefix.clone())
                        };

                    let mut found_paths: Vec<ExprPath> = Vec::new();
                    for term in &focus_terms {
                        let paths_before = found_paths.len();

                        // V2.14.32: Search within the scoped subtree only
                        // This limits highlighting to the focused area, not ALL occurrences globally
                        for sub_path in
                            diff_find_all_paths_to_expr(self.context, search_scope, *term)
                        {
                            // Prepend scope_path_prefix to get the full path from root
                            let mut full_path = scope_path_prefix.clone();
                            full_path.extend(sub_path.clone());
                            // Avoid duplicate paths
                            if !found_paths.contains(&full_path) {
                                found_paths.push(full_path);
                            }
                        }

                        // V2.9.24: If ExprId-based search found nothing for THIS term,
                        // try structural search. This handles dynamically constructed terms
                        // (e.g., from inverse_trig rules) where ExprIds differ but
                        // expressions are structurally equivalent.
                        if found_paths.len() == paths_before {
                            for sub_path in
                                diff_find_paths_by_structure(self.context, search_scope, *term)
                            {
                                let mut full_path = scope_path_prefix.clone();
                                full_path.extend(sub_path.clone());
                                if !found_paths.contains(&full_path) {
                                    found_paths.push(full_path);
                                }
                            }
                        }
                    }

                    let before = if !found_paths.is_empty() {
                        // Use path-based multi-term highlighting for accuracy
                        let mut before_config = PathHighlightConfig::new();
                        for path in found_paths {
                            before_config.add(path, HighlightColor::Red);
                        }
                        PathHighlightedLatexRenderer {
                            context: self.context,
                            id: global_before_expr,
                            path_highlights: &before_config,
                            hints: Some(&display_hints),
                            style_prefs: Some(&self.style_prefs),
                        }
                        .to_latex()
                    } else {
                        // Fallback: use step.path if no paths found to individual terms
                        let expr_path = pathsteps_to_expr_path(&step.path);
                        let mut before_config = PathHighlightConfig::new();
                        before_config.add(expr_path, HighlightColor::Red);
                        PathHighlightedLatexRenderer {
                            context: self.context,
                            id: global_before_expr,
                            path_highlights: &before_config,
                            hints: Some(&display_hints),
                            style_prefs: Some(&self.style_prefs),
                        }
                        .to_latex()
                    };
                    // AFTER: Try to find path to focus_after in global_after_expr
                    // This handles cases where tree reordering changes the structure
                    let after = if let Some(after_path) =
                        diff_find_path_to_expr(self.context, global_after_expr, focus_after)
                    {
                        // Found path - use path-based highlighting for accuracy
                        let mut after_config = PathHighlightConfig::new();
                        after_config.add(after_path, HighlightColor::Green);
                        PathHighlightedLatexRenderer {
                            context: self.context,
                            id: global_after_expr,
                            path_highlights: &after_config,
                            hints: Some(&display_hints),
                            style_prefs: Some(&self.style_prefs),
                        }
                        .to_latex()
                    } else {
                        // Path not found - fall back to ExprId-based highlighting
                        let mut after_config = HighlightConfig::new();
                        after_config.add(focus_after, HighlightColor::Green);
                        LaTeXExprHighlighted {
                            context: self.context,
                            id: global_after_expr,
                            highlights: &after_config,
                        }
                        .to_latex()
                    };

                    (before, after)
                }
            } else {
                // Standard case: use step.path for highlighting
                let expr_path = pathsteps_to_expr_path(&step.path);
                let mut before_config = PathHighlightConfig::new();
                before_config.add(expr_path.clone(), HighlightColor::Red);
                let before = PathHighlightedLatexRenderer {
                    context: self.context,
                    id: global_before_expr,
                    path_highlights: &before_config,
                    hints: Some(&display_hints),
                    style_prefs: Some(&self.style_prefs),
                }
                .to_latex();

                let mut after_config = PathHighlightConfig::new();
                after_config.add(expr_path, HighlightColor::Green);
                let after = PathHighlightedLatexRenderer {
                    context: self.context,
                    id: global_after_expr,
                    path_highlights: &after_config,
                    hints: Some(&display_hints),
                    style_prefs: Some(&self.style_prefs),
                }
                .to_latex();

                (before, after)
            };

            // Note: We intentionally do NOT skip steps where LaTeX renders identically.
            // The LaTeX renderer normalizes expressions (e.g., 1*x → x), which would
            // incorrectly filter Identity Property steps. Upstream to_display_steps
            // already removes structural no-ops (before == after ExprId).

            // Generate colored rule display: red antecedent → green consequent
            // Use before_local/after_local (Focus) if available, otherwise fall back to before/after
            let focus_before = step.before_local.unwrap_or(step.before);
            let focus_after = step.after_local.unwrap_or(step.after);

            let mut rule_before_config = HighlightConfig::new();
            rule_before_config.add(focus_before, HighlightColor::Red);
            let local_before_colored = cas_ast::LaTeXExprHighlightedWithHints {
                context: self.context,
                id: focus_before,
                highlights: &rule_before_config,
                hints: &display_hints,
                style_prefs: Some(&self.style_prefs),
            }
            .to_latex();

            let mut rule_after_config = HighlightConfig::new();
            rule_after_config.add(focus_after, HighlightColor::Green);
            let local_after_colored = cas_ast::LaTeXExprHighlightedWithHints {
                context: self.context,
                id: focus_after,
                highlights: &rule_after_config,
                hints: &display_hints,
                style_prefs: Some(&self.style_prefs),
            }
            .to_latex();

            let local_change_latex = format!(
                "{} \\rightarrow {}",
                local_before_colored, local_after_colored
            );

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
                            if !sub.before_expr.is_empty() {
                                details_html.push_str(&format!(
                                    r#"<div class="substep-math">\[{} \rightarrow {}\]</div>"#,
                                    sub.before_expr, sub.after_expr
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

            // V2.14.45: Build HTML for rule-provided substeps (educational explanations)
            let rule_substeps_html = if !step.substeps.is_empty() {
                let mut details_html = String::from(
                    r#"<details class="substeps-details" open>
                    <summary>Pasos didácticos</summary>
                    <div class="substeps-content">"#,
                );
                for substep in &step.substeps {
                    details_html.push_str(&format!(
                        r#"<div class="substep">
                            <strong>[{}]</strong>"#,
                        html_escape(&substep.title)
                    ));
                    for line in &substep.lines {
                        details_html.push_str(&format!(
                            r#"<div class="substep-line">• {}</div>"#,
                            html_escape(line)
                        ));
                    }
                    details_html.push_str("</div>");
                }
                details_html.push_str("</div></details>");
                details_html
            } else {
                String::new()
            };

            // V2.12.13: Build assumption HTML from assumption_events, filtered and grouped by kind
            let domain_html = if !step.assumption_events.is_empty() {
                use crate::assumptions::AssumptionKind;

                // Filter to displayable events only
                let displayable: Vec<_> = step
                    .assumption_events
                    .iter()
                    .filter(|e| e.kind.should_display())
                    .collect();

                if displayable.is_empty() {
                    String::new()
                } else {
                    let mut parts = Vec::new();

                    // Group by kind and format with icons
                    let requires: Vec<_> = displayable
                        .iter()
                        .filter(|e| matches!(e.kind, AssumptionKind::RequiresIntroduced))
                        .map(|e| html_escape(&e.message))
                        .collect();
                    if !requires.is_empty() {
                        parts.push(format!("ℹ️ Requires: {}", requires.join(", ")));
                    }

                    let branches: Vec<_> = displayable
                        .iter()
                        .filter(|e| matches!(e.kind, AssumptionKind::BranchChoice))
                        .map(|e| html_escape(&e.message))
                        .collect();
                    if !branches.is_empty() {
                        parts.push(format!("🔀 Branch: {}", branches.join(", ")));
                    }

                    let domain_ext: Vec<_> = displayable
                        .iter()
                        .filter(|e| matches!(e.kind, AssumptionKind::DomainExtension))
                        .map(|e| html_escape(&e.message))
                        .collect();
                    if !domain_ext.is_empty() {
                        parts.push(format!("🧿 Domain: {}", domain_ext.join(", ")));
                    }

                    let assumes: Vec<_> = displayable
                        .iter()
                        .filter(|e| matches!(e.kind, AssumptionKind::HeuristicAssumption))
                        .map(|e| html_escape(&e.message))
                        .collect();
                    if !assumes.is_empty() {
                        parts.push(format!("⚠️ Assumes: {}", assumes.join(", ")));
                    }

                    if parts.is_empty() {
                        String::new()
                    } else {
                        format!(
                            r#"                    <div class="domain-assumptions">{}</div>
"#,
                            parts.join("<br/>")
                        )
                    }
                }
            } else {
                String::new()
            };

            // V2.12.13: Per-step requires removed - they are now shown once in the
            // global-requires section at the end of the timeline. This avoids redundancy
            // when the same conditions appear on multiple steps.
            let requires_html = String::new();

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
                    {}
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
                rule_substeps_html, // Add rule-provided educational substeps
                global_after,
                requires_html,
                domain_html
            ));
        }

        // Add final result with display hints for consistent root notation
        // Use simplified_result if available (passed from simplifier), otherwise use last_global_after
        let final_result_expr = self.simplified_result.unwrap_or(last_global_after);

        // Check if result is a poly_result - render as text (not LaTeX) for large polynomials
        if let Some(poly_text) =
            crate::poly_store::try_render_poly_result(self.context, final_result_expr)
        {
            // Get term count for info badge
            let term_count = poly_text.matches('+').count() + 1;
            html.push_str(
                r#"        </div>
        <div class="final-result">
            <strong>🧮 Final Result</strong> <span class="poly-badge">"#,
            );
            html.push_str(&format!("Polynomial: {} terms", term_count));
            html.push_str(
                r#"</span>
            <pre class="poly-output">"#,
            );
            html.push_str(&html_escape(&poly_text));
            html.push_str(
                r#"</pre>
        </div>
"#,
            );
        } else {
            // Standard LaTeX rendering for normal expressions
            // V2.14.40: Use styled renderer for consistent root notation
            let empty_config = PathHighlightConfig::new();
            let final_expr = PathHighlightedLatexRenderer {
                context: self.context,
                id: final_result_expr,
                path_highlights: &empty_config,
                hints: Some(&display_hints),
                style_prefs: Some(&self.style_prefs),
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
"#,
            );
        }

        // V2.12.13: Add global requires section (inferred from input expression)
        // This ensures timeline shows the same requires as REPL
        if !self.global_requires.is_empty() {
            let requires_messages = crate::implicit_domain::render_conditions_normalized(
                self.context,
                &self.global_requires,
            );
            if !requires_messages.is_empty() {
                html.push_str(r#"        <div class="global-requires">"#);
                html.push_str("\n            <strong>ℹ️ Requires:</strong> ");
                let escaped: Vec<String> =
                    requires_messages.iter().map(|s| html_escape(s)).collect();
                html.push_str(&escaped.join(", "));
                html.push_str("\n        </div>\n");
            }
        }

        html.push_str(
            r#"    </div>
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

        // Clean up identity patterns like "\cdot 1" for better display
        clean_latex_identities(&html)
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
        max-width: 95%;
        margin: 0 auto;
        padding: 20px 10px;
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
        overflow-x: auto;
        max-width: 100%;
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
    /* Collapsible substeps styles */
    .substeps-toggle {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 12px;
        padding: 8px 12px;
        background: rgba(100, 181, 246, 0.15);
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.9em;
        color: var(--step-number-color);
        border: 1px solid rgba(100, 181, 246, 0.3);
        transition: all 0.2s ease;
    }}
    .substeps-toggle:hover {{
        background: rgba(100, 181, 246, 0.25);
    }}
    .substeps-toggle .arrow {{
        transition: transform 0.3s ease;
        font-size: 0.8em;
    }}
    .substeps-toggle.expanded .arrow {{
        transform: rotate(90deg);
    }}
    .substeps-container {{
        display: none;
        margin-top: 12px;
        padding-left: 20px;
        border-left: 2px solid rgba(100, 181, 246, 0.3);
    }}
    .substeps-container.visible {{
        display: block;
        animation: slideDown 0.3s ease;
    }}
    @keyframes slideDown {{
        from {{ opacity: 0; transform: translateY(-10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .substep {{
        background: rgba(30, 40, 55, 0.6);
        border-radius: 8px;
        padding: 12px 15px;
        margin-bottom: 10px;
        border-left: 3px solid #90caf9;
    }}
    .substep-number {{
        color: #90caf9;
        font-weight: bold;
        font-size: 0.85em;
        margin-bottom: 4px;
    }}
    .substep-description {{
        color: var(--description-color);
        font-size: 0.9em;
        margin-bottom: 8px;
        font-style: italic;
    }}
    .substep-equation {{
        background: rgba(20, 30, 45, 0.8);
        padding: 10px;
        border-radius: 6px;
        text-align: center;
        font-size: 1em;
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

            // Start the step div
            html.push_str(&format!(
                r#"        <div class="step">
            <div class="step-number">Step {}</div>
            <div class="description">{}</div>
            <div class="equation">
                \[{}\]
            </div>
"#,
                step_number,
                html_escape(&step.description),
                eq_latex
            ));

            // Add collapsible substeps if any
            if !step.substeps.is_empty() {
                let substep_id = format!("substeps-{}", step_number);
                html.push_str(&format!(
                    r#"            <div class="substeps-toggle" onclick="toggleSubsteps('{}')">
                <span class="arrow">▶</span>
                <span>Show derivation ({} steps)</span>
            </div>
            <div id="{}" class="substeps-container">
"#,
                    substep_id,
                    step.substeps.len(),
                    substep_id
                ));

                for (j, substep) in step.substeps.iter().enumerate() {
                    let sub_eq_latex = format!(
                        "{} {} {}",
                        LaTeXExpr {
                            context: self.context,
                            id: substep.equation_after.lhs
                        }
                        .to_latex(),
                        self.relop_to_latex(&substep.equation_after.op),
                        LaTeXExpr {
                            context: self.context,
                            id: substep.equation_after.rhs
                        }
                        .to_latex()
                    );

                    html.push_str(&format!(
                        r#"                <div class="substep">
                    <div class="substep-number">Step {}.{}</div>
                    <div class="substep-description">{}</div>
                    <div class="substep-equation">
                        \[{}\]
                    </div>
                </div>
"#,
                        step_number,
                        j + 1,
                        html_escape(&substep.description),
                        sub_eq_latex
                    ));
                }

                html.push_str("            </div>\n");
            }

            html.push_str("        </div>\n");

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
                // V2.x: Skip "otherwise" cases that only contain Residual (not useful info)
                let case_strs: Vec<String> = cases
                    .iter()
                    .filter_map(|case| {
                        // Skip "otherwise" cases that only contain Residual
                        if case.when.is_otherwise()
                            && matches!(&case.then.solutions, SolutionSet::Residual(_))
                        {
                            return None;
                        }
                        let sol_latex = self.solution_set_inner_to_latex(&case.then.solutions);
                        if case.when.is_otherwise() {
                            Some(format!("{} & \\text{{otherwise}}", sol_latex))
                        } else {
                            let cond_latex = case.when.latex_display_with_context(self.context);
                            Some(format!("{} & \\text{{if }} {}", sol_latex, cond_latex))
                        }
                    })
                    .collect();
                // If only one case remains after filtering, render without \begin{cases}
                if case_strs.len() == 1 {
                    // Extract just the solution part (before the " & \text{if}")
                    let single = &case_strs[0];
                    if let Some(idx) = single.find(r" & \text{if}") {
                        return single[..idx].to_string();
                    }
                }
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
        r#"    <script>
        function toggleSubsteps(id) {
            const container = document.getElementById(id);
            const toggle = document.querySelector(`[onclick*="${id}"]`);
            if (container.classList.contains('visible')) {
                container.classList.remove('visible');
                toggle.classList.remove('expanded');
                toggle.querySelector('span:last-child').textContent = 'Show derivation (' + container.children.length + ' steps)';
            } else {
                container.classList.add('visible');
                toggle.classList.add('expanded');
                toggle.querySelector('span:last-child').textContent = 'Hide derivation';
                // Re-render MathJax for the newly visible content
                if (window.MathJax) {
                    MathJax.typeset([container]);
                }
            }
        }
    </script>
    <footer>
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
