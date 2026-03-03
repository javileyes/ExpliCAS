//! Didactic Step Enhancement Layer
//!
//! This module provides visualization-layer enrichment of engine steps
//! without modifying the core engine. It post-processes steps to add
//! instructive detail for human learners.
//!
//! # Architecture
//! - Pure post-processing: never modifies engine behavior
//! - Optional: can be enabled/disabled via verbosity
//! - Extensible: easy to add new enrichers
//!
//! # Contract (V2.12.13)
//!
//! **SubSteps explain techniques within a Step. They MUST NOT duplicate
//! decompositions that already exist as chained Steps via ChainedRewrite.**
//!
//! When `step.is_chained == true`, the Step was created from a ChainedRewrite
//! and already has proper before/after expressions. Skip substep generation
//! that would duplicate this information (e.g., GCD factorization substeps).
//!
//! # When to use which:
//!
//! - **ChainedRewrite**: Multi-step algebraic decomposition with real ExprIds
//!   (e.g., Factor → Cancel as separate visible Steps)
//! - **SubSteps**: Educational annotation explaining technique (e.g., "Find conjugate")
//!
//! # Example
//! ```ignore
//! let enriched = didactic::enrich_steps(&ctx, original_expr, steps);
//! for step in enriched {
//!     println!("{}", step.base_step.description);
//!     for sub in &step.sub_steps {
//!         println!("    → {}", sub.description);
//!     }
//! }
//! ```

mod fraction_steps;
mod nested_fractions;

use cas_ast::{Context, Expr, ExprId};
use cas_solver::{AssumptionEvent, ImportanceLevel, PathStep, Step};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

use fraction_steps::{
    detect_exponent_fraction_change, find_all_fraction_sums, generate_fraction_sum_substeps,
    generate_gcd_factorization_substeps,
};
use nested_fractions::{
    generate_nested_fraction_substeps, generate_polynomial_identity_substeps,
    generate_rationalization_substeps, generate_root_denesting_substeps,
    generate_sum_three_cubes_substeps,
};

/// Display verbosity mode for didactic simplification rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepDisplayMode {
    None,
    Succinct,
    Normal,
    Verbose,
}

/// An enriched step with optional sub-steps for didactic explanation
#[derive(Debug, Clone)]
pub struct EnrichedStep {
    /// The original step from the engine
    pub base_step: Step,
    /// Synthetic sub-steps that explain hidden operations
    pub sub_steps: Vec<SubStep>,
}

/// A synthetic sub-step that explains a hidden operation
#[derive(Debug, Clone)]
pub struct SubStep {
    /// Human-readable description of the operation
    pub description: String,
    /// Expression before the operation (plain text for CLI display)
    pub before_expr: String,
    /// Expression after the operation (plain text for CLI display)
    pub after_expr: String,
    /// Optional LaTeX for `before_expr` (for web/MathJax rendering).
    /// When set, the JSON layer uses this instead of `before_expr`.
    pub before_latex: Option<String>,
    /// Optional LaTeX for `after_expr` (for web/MathJax rendering).
    /// When set, the JSON layer uses this instead of `after_expr`.
    pub after_latex: Option<String>,
}

impl SubStep {
    /// Create a plain-text sub-step (no LaTeX).
    /// Text will be wrapped in `\text{}` by the JSON layer.
    pub fn new(
        description: impl Into<String>,
        before_expr: impl Into<String>,
        after_expr: impl Into<String>,
    ) -> Self {
        Self {
            description: description.into(),
            before_expr: before_expr.into(),
            after_expr: after_expr.into(),
            before_latex: None,
            after_latex: None,
        }
    }

    /// Set the LaTeX for `before_expr`.
    pub fn with_before_latex(mut self, latex: impl Into<String>) -> Self {
        self.before_latex = Some(latex.into());
        self
    }

    /// Set the LaTeX for `after_expr`.
    pub fn with_after_latex(mut self, latex: impl Into<String>) -> Self {
        self.after_latex = Some(latex.into());
        self
    }
}

/// Enrich a list of steps with didactic sub-steps
///
/// This is the main entry point for the didactic layer.
/// It analyzes each step and adds explanatory sub-steps where helpful.
pub fn enrich_steps(ctx: &Context, original_expr: ExprId, steps: Vec<Step>) -> Vec<EnrichedStep> {
    let mut enriched = Vec::with_capacity(steps.len());

    // Check original expression for fraction sums (before any simplification)
    let all_fraction_sums = find_all_fraction_sums(ctx, original_expr);

    // Keep only the sum with the most fractions (ignore partial subsums)
    // AND deduplicate identical fraction sums
    let unique_fraction_sums: Vec<_> = if all_fraction_sums.is_empty() {
        Vec::new()
    } else {
        let max_fractions = all_fraction_sums
            .iter()
            .map(|s| s.fractions.len())
            .max()
            .unwrap_or(0);
        let mut seen = std::collections::HashSet::new();
        all_fraction_sums
            .into_iter()
            .filter(|info| info.fractions.len() == max_fractions)
            .filter(|info| {
                // Deduplicate by result value
                let key = format!("{}", info.result);
                seen.insert(key)
            })
            .collect()
    };

    for (step_idx, step) in steps.iter().enumerate() {
        let mut sub_steps = Vec::new();

        // Attach fraction sum sub-steps to EVERY step
        // The CLI will track and show them only once on the first VISIBLE step
        // This ensures sub-steps appear even if early steps are filtered out
        if !unique_fraction_sums.is_empty() {
            for info in &unique_fraction_sums {
                sub_steps.extend(generate_fraction_sum_substeps(info));
            }
        }

        // Also check for fraction sums in exponent (between steps)
        if let Some(fraction_info) = detect_exponent_fraction_change(ctx, &steps, step_idx) {
            // Avoid duplicates
            if !unique_fraction_sums
                .iter()
                .any(|o| o.fractions == fraction_info.fractions)
            {
                sub_steps.extend(generate_fraction_sum_substeps(&fraction_info));
            }
        }

        // Add factorization sub-steps for fraction GCD simplification
        // V2.12.13: Gate by is_chained - if this step came from ChainedRewrite,
        // the Factor→Cancel decomposition already exists as separate Steps
        if step.description.starts_with("Simplified fraction by GCD") && !step.is_chained() {
            sub_steps.extend(generate_gcd_factorization_substeps(ctx, step));
        }

        // Add sub-steps for nested fraction simplification
        // Match by rule_name pattern (more stable than description string)
        let is_nested_fraction = step.rule_name.to_lowercase().contains("complex fraction")
            || step.rule_name.to_lowercase().contains("nested fraction")
            || step.description.to_lowercase().contains("nested fraction");
        if is_nested_fraction {
            sub_steps.extend(generate_nested_fraction_substeps(ctx, step));
        }

        // Add sub-steps for rationalization (generalized and product)
        if step.description.contains("Rationalize") || step.rule_name.contains("Rationalize") {
            sub_steps.extend(generate_rationalization_substeps(ctx, step));
        }

        // Add sub-steps for polynomial identity normalization (PolyZero airbag)
        if step.poly_proof().is_some() {
            sub_steps.extend(generate_polynomial_identity_substeps(ctx, step));
        }

        // Add sub-steps for Sum of Three Cubes identity
        if step.rule_name.contains("Sum of Three Cubes") {
            sub_steps.extend(generate_sum_three_cubes_substeps(ctx, step));
        }

        // Add sub-steps for Root Denesting
        if step.rule_name.contains("Root Denesting") {
            sub_steps.extend(generate_root_denesting_substeps(ctx, step));
        }

        enriched.push(EnrichedStep {
            base_step: step.clone(),
            sub_steps,
        });
    }

    enriched
}

/// Get didactic sub-steps for an expression when there are no simplification steps
///
/// This is useful when fraction sums are computed during parsing/canonicalization
/// and there are no engine steps to attach the explanation to.
///
/// Example: `x^(1/3 + 1/6)` becomes `x^(1/2)` without any steps,
/// but we want to show how 1/3 + 1/6 = 1/2.
pub fn get_standalone_substeps(ctx: &Context, original_expr: ExprId) -> Vec<SubStep> {
    let all_fraction_sums = find_all_fraction_sums(ctx, original_expr);

    if all_fraction_sums.is_empty() {
        return Vec::new();
    }

    // Keep only the sum with the most fractions (ignore partial subsums)
    let max_fractions = all_fraction_sums
        .iter()
        .map(|s| s.fractions.len())
        .max()
        .unwrap_or(0);
    let mut seen = std::collections::HashSet::new();
    let unique_fraction_sums: Vec<_> = all_fraction_sums
        .into_iter()
        .filter(|info| info.fractions.len() == max_fractions)
        .filter(|info| {
            let key = format!("{}", info.result);
            seen.insert(key)
        })
        .collect();

    let mut sub_steps = Vec::new();
    for info in &unique_fraction_sums {
        sub_steps.extend(generate_fraction_sum_substeps(info));
    }
    sub_steps
}

/// Rendering hints for CLI sub-step blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CliSubstepsRenderPlan {
    /// Optional category header to display before sub-steps.
    pub header: Option<&'static str>,
    /// If true, this block should be shown only once (deduplicated across steps).
    pub dedupe_once: bool,
}

/// Build a CLI rendering plan for enriched sub-steps.
///
/// This preserves the legacy precedence and dedupe behavior from REPL:
/// - Polynomial normalization header has highest priority.
/// - Fraction-sum header comes next (unless nested-fraction classification overrides it).
/// - Factorization header next.
/// - Nested-fraction header next.
/// - Fraction-sum-only blocks are deduplicated (shown once).
pub fn build_cli_substeps_render_plan(sub_steps: &[SubStep]) -> CliSubstepsRenderPlan {
    let has_fraction_sum = sub_steps.iter().any(|s| {
        s.description.contains("common denominator") || s.description.contains("Sum the fractions")
    });
    let has_factorization = sub_steps.iter().any(|s| {
        s.description.contains("Cancel common factor") || s.description.contains("Factor")
    });
    let has_nested_fraction = sub_steps.iter().any(|s| {
        s.description.contains("Combinar términos")
            || s.description.contains("Invertir la fracción")
            || s.description.contains("denominadores internos")
    });
    let has_polynomial_identity = sub_steps.iter().any(|s| {
        s.description.contains("forma normal polinómica")
            || s.description.contains("Cancelar términos semejantes")
    });

    let dedupe_once =
        has_fraction_sum && !has_nested_fraction && !has_factorization && !has_polynomial_identity;

    let header = if has_polynomial_identity {
        Some("[Normalización polinómica]")
    } else if has_fraction_sum && !has_nested_fraction {
        Some("[Suma de fracciones en exponentes]")
    } else if has_factorization {
        Some("[Factorización de polinomios]")
    } else if has_nested_fraction {
        Some("[Simplificación de fracción compleja]")
    } else {
        None
    };

    CliSubstepsRenderPlan {
        header,
        dedupe_once,
    }
}

/// Convert LaTeX-like notation into plain-text form for CLI display.
pub fn latex_to_plain_text(s: &str) -> String {
    let mut result = s.to_string();

    // Replace \cdot with multiplication dot.
    result = result.replace("\\cdot", " · ");

    // Replace \text{...} with its inner content.
    while let Some(start) = result.find("\\text{") {
        if let Some(end) = result[start + 6..].find('}') {
            let content = &result[start + 6..start + 6 + end];
            result = format!(
                "{}{}{}",
                &result[..start],
                content,
                &result[start + 7 + end..]
            );
        } else {
            break;
        }
    }

    // Recursively replace \frac{num}{den} with (num/den).
    let mut iterations = 0;
    while result.contains("\\frac{") && iterations < 10 {
        iterations += 1;
        if let Some(start) = result.rfind("\\frac{") {
            let rest = &result[start + 5..];
            if let Some((numer, numer_end)) = find_balanced_braces(rest) {
                let after_numer = &rest[numer_end + 1..];
                if after_numer.starts_with('{') {
                    if let Some((denom, denom_end)) = find_balanced_braces(after_numer) {
                        let total_end = start + 5 + numer_end + 1 + denom_end + 1;
                        let replacement = format!("({}/{})", numer, denom);
                        result = format!(
                            "{}{}{}",
                            &result[..start],
                            replacement,
                            &result[total_end..]
                        );
                        continue;
                    }
                }
            }
        }
        break;
    }

    result.replace("\\", "")
}

fn should_show_simplify_step(step: &Step, mode: StepDisplayMode) -> bool {
    match mode {
        StepDisplayMode::None => false,
        StepDisplayMode::Verbose => true,
        StepDisplayMode::Succinct | StepDisplayMode::Normal => {
            if step.get_importance() < ImportanceLevel::Medium {
                return false;
            }
            if let (Some(before), Some(after)) = (step.global_before, step.global_after) {
                if before == after {
                    return false;
                }
            }
            true
        }
    }
}

/// Format simplification steps for CLI/REPL text output.
///
/// This keeps didactic rendering rules outside frontends so clients can remain
/// thin and only handle I/O.
pub fn format_cli_simplification_steps(
    ctx: &mut Context,
    expr: ExprId,
    steps: &[Step],
    style_signals: cas_formatter::root_style::ParseStyleSignals,
    display_mode: StepDisplayMode,
) -> Vec<String> {
    use cas_formatter::root_style::StylePreferences;
    use cas_formatter::DisplayExprStyled;

    if display_mode == StepDisplayMode::None {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let style_prefs =
        StylePreferences::from_expression_with_signals(ctx, expr, Some(&style_signals));

    if steps.is_empty() {
        let standalone_substeps = get_standalone_substeps(ctx, expr);

        if !standalone_substeps.is_empty() && display_mode != StepDisplayMode::Succinct {
            lines.push("Computation:".to_string());
            for sub in &standalone_substeps {
                lines.push(format!("   → {}", sub.description));
                if !sub.before_expr.is_empty() {
                    lines.push(format!(
                        "     {} → {}",
                        latex_to_plain_text(&sub.before_expr),
                        latex_to_plain_text(&sub.after_expr)
                    ));
                }
            }
        } else if display_mode != StepDisplayMode::Succinct {
            lines.push("No simplification steps needed.".to_string());
        }

        return lines;
    }

    if display_mode != StepDisplayMode::Succinct {
        lines.push("Steps:".to_string());
    }

    let enriched_steps = enrich_steps(ctx, expr, steps.to_vec());
    let mut current_root = expr;
    let mut step_count = 0;
    let mut sub_steps_shown = false;

    for (step_idx, step) in steps.iter().enumerate() {
        if should_show_simplify_step(step, display_mode) {
            let before_disp = cas_formatter::clean_display_string(&format!(
                "{}",
                DisplayExprStyled::new(ctx, step.before, &style_prefs)
            ));
            let after_disp = cas_formatter::clean_display_string(&format!(
                "{}",
                DisplayExprStyled::new(ctx, step.after, &style_prefs)
            ));

            if before_disp == after_disp {
                if let Some(global_after) = step.global_after {
                    current_root = global_after;
                }
                continue;
            }

            step_count += 1;

            if display_mode == StepDisplayMode::Succinct {
                current_root = reconstruct_global_expr(ctx, current_root, step.path(), step.after);
                lines.push(format!(
                    "-> {}",
                    DisplayExprStyled::new(ctx, current_root, &style_prefs)
                ));
                continue;
            }

            lines.push(format!(
                "{}. {}  [{}]",
                step_count, step.description, step.rule_name
            ));

            if let Some(global_before) = step.global_before {
                lines.push(format!(
                    "   Before: {}",
                    cas_formatter::clean_display_string(&format!(
                        "{}",
                        DisplayExprStyled::new(ctx, global_before, &style_prefs)
                    ))
                ));
            } else {
                lines.push(format!(
                    "   Before: {}",
                    cas_formatter::clean_display_string(&format!(
                        "{}",
                        DisplayExprStyled::new(ctx, current_root, &style_prefs)
                    ))
                ));
            }

            if let Some(enriched_step) = enriched_steps.get(step_idx) {
                if !enriched_step.sub_steps.is_empty() {
                    let render_plan = build_cli_substeps_render_plan(&enriched_step.sub_steps);
                    let should_show = if render_plan.dedupe_once {
                        !sub_steps_shown
                    } else {
                        true
                    };

                    if should_show {
                        if render_plan.dedupe_once {
                            sub_steps_shown = true;
                        }
                        if let Some(header) = render_plan.header {
                            lines.push(format!("   {}", header));
                        }
                        for sub in &enriched_step.sub_steps {
                            lines.push(format!("      → {}", sub.description));
                            if !sub.before_expr.is_empty() {
                                lines.push(format!(
                                    "        {} → {}",
                                    latex_to_plain_text(&sub.before_expr),
                                    latex_to_plain_text(&sub.after_expr)
                                ));
                            }
                        }
                    }
                }
            }

            let (rule_before_id, rule_after_id) = match (step.before_local(), step.after_local()) {
                (Some(bl), Some(al)) => (bl, al),
                _ => (step.before, step.after),
            };

            let before_disp = cas_formatter::clean_display_string(&format!(
                "{}",
                DisplayExprStyled::new(ctx, rule_before_id, &style_prefs)
            ));
            let after_disp =
                cas_formatter::clean_display_string(&cas_formatter::render_with_rule_scope(
                    ctx,
                    rule_after_id,
                    &step.rule_name,
                    &style_prefs,
                ));

            if before_disp == after_disp {
                if let Some(global_after) = step.global_after {
                    current_root = global_after;
                }
                continue;
            }

            lines.push(format!("   Rule: {} -> {}", before_disp, after_disp));

            if !step.substeps().is_empty() {
                for substep in step.substeps() {
                    lines.push(format!("   [{}]", substep.title));
                    for line in &substep.lines {
                        lines.push(format!("      • {}", line));
                    }
                }
            }

            if let Some(global_after) = step.global_after {
                current_root = global_after;
            } else {
                current_root = reconstruct_global_expr(ctx, current_root, step.path(), step.after);
            }

            lines.push(format!(
                "   After: {}",
                cas_formatter::clean_display_string(&format!(
                    "{}",
                    DisplayExprStyled::new(ctx, current_root, &style_prefs)
                ))
            ));

            let assumption_events = cas_solver::assumption_events_from_step(step);
            for assumption_line in format_displayable_assumption_lines(&assumption_events) {
                lines.push(format!("   {}", assumption_line));
            }
        } else if let Some(global_after) = step.global_after {
            current_root = global_after;
        } else {
            current_root = reconstruct_global_expr(ctx, current_root, step.path(), step.after);
        }
    }

    lines
}

/// Variant of [`format_cli_simplification_steps`] that accepts a simplifier.
///
/// Keeps REPL frontends from reaching into `simplifier.context` directly.
pub fn format_cli_simplification_steps_with_simplifier(
    simplifier: &mut cas_solver::Simplifier,
    expr: ExprId,
    steps: &[Step],
    style_signals: cas_formatter::root_style::ParseStyleSignals,
    display_mode: StepDisplayMode,
) -> Vec<String> {
    format_cli_simplification_steps(
        &mut simplifier.context,
        expr,
        steps,
        style_signals,
        display_mode,
    )
}

/// Extract content within balanced braces starting at the first `{`.
fn find_balanced_braces(s: &str) -> Option<(String, usize)> {
    let mut depth = 0;
    let mut content = String::new();
    for (i, c) in s.chars().enumerate() {
        match c {
            '{' => {
                if depth > 0 {
                    content.push(c);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((content, i));
                }
                content.push(c);
            }
            _ => {
                if depth > 0 {
                    content.push(c);
                }
            }
        }
    }
    None
}

fn format_displayable_assumption_lines(events: &[AssumptionEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| {
            if event.kind.should_display() {
                Some(format!(
                    "{} {}: {}",
                    event.kind.icon(),
                    event.kind.label(),
                    event.message
                ))
            } else {
                None
            }
        })
        .collect()
}

fn reconstruct_global_expr(
    context: &mut Context,
    root: ExprId,
    path: &[PathStep],
    replacement: ExprId,
) -> ExprId {
    if path.is_empty() {
        return replacement;
    }

    let current_step = &path[0];
    let remaining_path = &path[1..];
    let expr = context.get(root).clone();

    match (expr, current_step) {
        (Expr::Add(l, r), PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Add(new_l, r))
        }
        // Sub(a,b) may be canonicalized as Add(a, Neg(b)).
        (Expr::Add(l, r), PathStep::Right) => {
            if let Expr::Neg(inner) = context.get(r).clone() {
                let new_inner =
                    reconstruct_global_expr(context, inner, remaining_path, replacement);
                let new_neg = context.add(Expr::Neg(new_inner));
                context.add(Expr::Add(l, new_neg))
            } else {
                let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
                context.add(Expr::Add(l, new_r))
            }
        }
        (Expr::Sub(l, r), PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Sub(new_l, r))
        }
        (Expr::Sub(l, r), PathStep::Right) => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Sub(l, new_r))
        }
        (Expr::Mul(l, r), PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Mul(new_l, r))
        }
        (Expr::Mul(l, r), PathStep::Right) => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Mul(l, new_r))
        }
        (Expr::Div(l, r), PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Div(new_l, r))
        }
        (Expr::Div(l, r), PathStep::Right) => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Div(l, new_r))
        }
        (Expr::Pow(b, e), PathStep::Base) => {
            let new_b = reconstruct_global_expr(context, b, remaining_path, replacement);
            context.add(Expr::Pow(new_b, e))
        }
        (Expr::Pow(b, e), PathStep::Exponent) => {
            let new_e = reconstruct_global_expr(context, e, remaining_path, replacement);
            context.add(Expr::Pow(b, new_e))
        }
        (Expr::Neg(e), PathStep::Inner) => {
            let new_e = reconstruct_global_expr(context, e, remaining_path, replacement);
            context.add(Expr::Neg(new_e))
        }
        (Expr::Function(name, args), PathStep::Arg(idx)) => {
            let mut new_args = args.clone();
            if *idx < new_args.len() {
                new_args[*idx] =
                    reconstruct_global_expr(context, new_args[*idx], remaining_path, replacement);
                context.add(Expr::Function(name, new_args))
            } else {
                root
            }
        }
        _ => root,
    }
}

// --- Shared helpers used by submodules ---

/// Try to interpret an expression as a fraction (BigRational)
/// Handles both Number(n) and Div(Number, Number) patterns
fn try_as_fraction(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Div(numer, denom) => {
            // Check if both numerator and denominator are numbers
            if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*numer), ctx.get(*denom)) {
                // Convert to BigRational: n/d
                if !d.is_zero() {
                    // n and d are already BigRational, compute n/d
                    return Some(n / d);
                }
            }
            None
        }
        _ => None,
    }
}

/// Collect all terms from an Add chain
fn collect_add_terms(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_add_terms(ctx, *l, terms);
            collect_add_terms(ctx, *r, terms);
        }
        _ => terms.push(expr),
    }
}

/// Format a BigRational as a LaTeX fraction or integer
fn format_fraction(r: &BigRational) -> String {
    if r.denom().is_one() {
        format!("{}", r.numer())
    } else {
        format!("\\frac{{{}}}{{{}}}", r.numer(), r.denom())
    }
}

/// Compute LCM of two BigInts
fn lcm_bigint(a: &BigInt, b: &BigInt) -> BigInt {
    if a.is_zero() || b.is_zero() {
        BigInt::zero()
    } else {
        (a * b).abs() / gcd_bigint(a, b)
    }
}

/// Compute GCD of two BigInts using Euclidean algorithm
fn gcd_bigint(a: &BigInt, b: &BigInt) -> BigInt {
    let mut a = a.abs();
    let mut b = b.abs();
    while !b.is_zero() {
        let temp = b.clone();
        b = &a % &b;
        a = temp;
    }
    a
}

// Trait for is_one check on BigInt
trait IsOne {
    fn is_one(&self) -> bool;
}

impl IsOne for BigInt {
    fn is_one(&self) -> bool {
        *self == BigInt::from(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_math::expr_predicates::contains_division_like_term;
    use fraction_steps::FractionSumInfo;
    use nested_fractions::{
        classify_nested_fraction, extract_combined_fraction_str, NestedFractionPattern,
    };

    #[test]
    fn test_format_fraction() {
        let half = BigRational::new(BigInt::from(1), BigInt::from(2));
        assert_eq!(format_fraction(&half), "\\frac{1}{2}");

        let three = BigRational::from_integer(BigInt::from(3));
        assert_eq!(format_fraction(&three), "3");
    }

    #[test]
    fn test_gcd_lcm() {
        let a = BigInt::from(12);
        let b = BigInt::from(8);
        assert_eq!(gcd_bigint(&a, &b), BigInt::from(4));
        assert_eq!(lcm_bigint(&a, &b), BigInt::from(24));
    }

    #[test]
    fn test_fraction_sum_substeps() {
        let fractions = vec![
            BigRational::new(BigInt::from(1), BigInt::from(24)),
            BigRational::new(BigInt::from(1), BigInt::from(2)),
            BigRational::new(BigInt::from(1), BigInt::from(6)),
        ];
        let result: BigRational = fractions.iter().cloned().sum();

        let info = FractionSumInfo {
            fractions,
            result: result.clone(),
        };

        let substeps = generate_fraction_sum_substeps(&info);
        assert!(!substeps.is_empty());

        // Result should be 17/24
        assert_eq!(result, BigRational::new(BigInt::from(17), BigInt::from(24)));
    }

    #[test]
    fn test_nested_fraction_pattern_classification_p1() {
        // P1: 1/(1 + 1/x) - unit fraction in denominator
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));
        let one_over_x = ctx.add(Expr::Div(one, x));
        let denom = ctx.add(Expr::Add(one, one_over_x));
        let expr = ctx.add(Expr::Div(one, denom));

        let pattern = classify_nested_fraction(&ctx, expr);
        assert!(matches!(
            pattern,
            Some(NestedFractionPattern::OneOverSumWithUnitFraction)
        ));
    }

    #[test]
    fn test_nested_fraction_pattern_classification_p3() {
        // P3: 2/(1 + 1/x) - non-unit numerator
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));
        let two = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(2))));
        let one_over_x = ctx.add(Expr::Div(one, x));
        let denom = ctx.add(Expr::Add(one, one_over_x));
        let expr = ctx.add(Expr::Div(two, denom));

        let pattern = classify_nested_fraction(&ctx, expr);
        assert!(matches!(
            pattern,
            Some(NestedFractionPattern::FractionOverSumWithFraction)
        ));
    }

    #[test]
    fn test_extract_combined_fraction_simple() {
        // 1 + 1/x → "(1 · x + 1) / x"
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));
        let one_over_x = ctx.add(Expr::Div(one, x));
        let add_expr = ctx.add(Expr::Add(one, one_over_x));

        let result = extract_combined_fraction_str(&ctx, add_expr);
        assert!(
            result.contains("x"),
            "Should contain denominator 'x': {}",
            result
        );
        assert!(
            result.contains("1"),
            "Should contain numerator '1': {}",
            result
        );
    }

    #[test]
    fn test_extract_combined_fraction_complex_denominator() {
        // 1 + x/(x+1) → LaTeX format: \frac{1 \cdot (x + 1) + x}{x + 1}
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));
        let x_plus_1 = ctx.add(Expr::Add(x, one));
        let x_over_xplus1 = ctx.add(Expr::Div(x, x_plus_1));
        let add_expr = ctx.add(Expr::Add(one, x_over_xplus1));

        let result = extract_combined_fraction_str(&ctx, add_expr);
        // Should be LaTeX format with \frac
        assert!(
            result.contains("\\frac"),
            "Should contain LaTeX \\frac: {}",
            result
        );
        assert!(
            result.contains("\\cdot"),
            "Should contain LaTeX \\cdot for multiplication: {}",
            result
        );
    }

    #[test]
    fn test_contains_div_simple() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));

        // x does not contain div
        assert!(!contains_division_like_term(&ctx, x));

        // 1/x contains div
        let div = ctx.add(Expr::Div(one, x));
        assert!(contains_division_like_term(&ctx, div));

        // 1 + 1/x contains div
        let add = ctx.add(Expr::Add(one, div));
        assert!(contains_division_like_term(&ctx, add));
    }

    #[test]
    fn test_build_cli_substeps_render_plan_fraction_sum_deduped() {
        let sub_steps = vec![SubStep::new(
            "Find common denominator for fractions",
            "",
            "",
        )];
        let plan = build_cli_substeps_render_plan(&sub_steps);
        assert_eq!(plan.header, Some("[Suma de fracciones en exponentes]"));
        assert!(plan.dedupe_once);
    }

    #[test]
    fn test_latex_to_plain_text_converts_frac_and_text() {
        let input = r"\text{Paso}: \frac{1}{x+1} \cdot y";
        let output = latex_to_plain_text(input);
        assert!(output.contains("Paso"));
        assert!(output.contains("(1/x+1)"));
        assert!(output.contains("·"));
    }
}
