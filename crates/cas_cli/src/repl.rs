use cas_engine::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule, MulZeroRule};
use cas_engine::rules::exponents::{
    EvaluatePowerRule, IdentityPowerRule, PowerPowerRule, PowerProductRule, PowerQuotientRule,
    ProductPowerRule,
};
use cas_engine::rules::polynomial::{AnnihilationRule, CombineLikeTermsRule};
use cas_engine::Simplifier;

use cas_ast::{
    Context, DisplayExpr, DisplayExprStyled, Expr, ExprId, ParseStyleSignals, StylePreferences,
};
use cas_engine::rules::algebra::{ExpandRule, FactorRule, SimplifyFractionRule};
use cas_engine::rules::calculus::{DiffRule, IntegrateRule};
use cas_engine::rules::grouping::CollectRule;
use cas_engine::rules::logarithms::{EvaluateLogRule, ExponentialLogRule};
use cas_engine::rules::number_theory::NumberTheoryRule;
use cas_engine::rules::trigonometry::{
    AngleIdentityRule, DoubleAngleRule, EvaluateTrigRule, PythagoreanIdentityRule, TanToSinCosRule,
};
use cas_engine::step::PathStep;
use rustyline::error::ReadlineError;

use crate::completer::CasHelper;
use crate::config::CasConfig;

/// Unwrap top-level __hold() wrapper (used for eager let bindings)
fn unwrap_hold_top(ctx: &Context, expr: ExprId) -> ExprId {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if ctx.sym_name(*name) == "__hold" && args.len() == 1 {
            return args[0];
        }
    }
    expr
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    None,
    Succinct, // Compact: same filtering as Normal but 1 line per step
    Normal,
    Verbose,
}

/// Clean verbose patterns from display strings for better didactic quality
/// Removes patterns like "1 * x" -> "x" and "x * 1" -> "x"
fn clean_display_string(s: &str) -> String {
    let mut result = s.to_string();
    let mut changed = true;

    // Iterate until no more changes (handles nested patterns)
    while changed {
        let before = result.clone();

        // Replace "1 * " at the very start
        if result.starts_with("1 * ") {
            result = result[4..].to_string();
        }

        // Replace " * 1" at the very end (but not " * 10" etc)
        if result.ends_with(" * 1") && !result.ends_with("0 * 1") {
            result = result[..result.len() - 4].to_string();
        }

        // Order matters: do more specific patterns first

        // "(1 * " at start of parenthesized expression
        result = result.replace("(1 * ", "(");

        // " * 1)" at end of parenthesized expression
        result = result.replace(" * 1)", ")");

        // "1 * 1" -> "1" (common in fraction combination)
        result = result.replace("1 * 1", "1");

        // " + 1 * " -> " + " (after addition)
        result = result.replace(" + 1 * ", " + ");

        // " - 1 * " -> " - " (after subtraction)
        result = result.replace(" - 1 * ", " - ");

        // "/ (1 * " -> "/ ("
        result = result.replace("/ (1 * ", "/ (");

        // " * 1 +" -> " +"
        result = result.replace(" * 1 +", " +");

        // " * 1 -" -> " -"
        result = result.replace(" * 1 -", " -");

        // " * 1 /" -> " /"
        result = result.replace(" * 1 /", " /");

        // " * 1 *" -> " *" (chained multiplications)
        result = result.replace(" * 1 *", " *");

        // Handle edge case: "1 * -" at start (like "1 * -1^2")
        if result.starts_with("1 * -") {
            result = result[4..].to_string();
        }

        // "1 * " folsuccincted by digit at start (like "1 * 2")
        if result.starts_with("1 * ") && result.len() > 4 {
            let next_char = result.chars().nth(4);
            if let Some(c) = next_char {
                if c.is_ascii_digit() || c == '-' || c == 'x' || c == 'y' || c == '(' {
                    result = result[4..].to_string();
                }
            }
        }

        // "/ (1 * x)" -> "/ x" for denominators
        result = result.replace("/ (1 * x)", "/ x");
        result = result.replace("/ (1 * y)", "/ y");

        // Handle "2 * -1 * x" -> "-2 * x" is too complex, instead handle "* -1 * " later
        result = result.replace(" * -1 * ", " * -");

        // Also handle middot (·) patterns used in display output
        // "1·x" -> "x" and "x·1" -> "x"
        result = result.replace("(1·", "(");
        result = result.replace("·1)", ")");
        result = result.replace(" + 1·", " + ");
        result = result.replace(" - 1·", " - ");
        result = result.replace("·1 +", " +");
        result = result.replace("·1 -", " -");
        result = result.replace("·1 /", " /");
        result = result.replace("·1·", "·");
        result = result.replace("·1)", ")");
        // Handle end of string patterns
        if result.ends_with("·1") {
            result = result[..result.len() - 3].to_string(); // ·1 is 3 bytes in UTF-8
        }
        // Handle start of string
        if result.starts_with("1·") && result.len() > 3 {
            result = result[3..].to_string(); // 1· is 3 bytes
        }

        changed = before != result;
    }

    // Remove __hold(...) wrapper - it's an internal invisible barrier
    while let Some(start) = result.find("__hold(") {
        // Find matching closing paren
        let content_start = start + 7; // Length of "__hold("
        let mut depth = 1;
        let mut end = content_start;
        for (i, c) in result[content_start..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = content_start + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth == 0 {
            let inner = &result[content_start..end];
            result = format!("{}{}{}", &result[..start], inner, &result[end + 1..]);
        } else {
            break; // Malformed, stop
        }
    }

    // Clean sign patterns (matches LaTeX cleanup logic)
    clean_sign_patterns(result)
}

/// Clean sign patterns from display strings.
/// Converts `+ -` to `-` and `- -` to `+` only when followed by a digit or variable,
/// NOT when followed by `(` (which indicates a grouped subexpression like `(-1)²`).
/// This mirrors the LaTeX cleanup logic in clean_latex_negatives.
fn clean_sign_patterns(s: String) -> String {
    use regex::Regex;
    let mut result = s;

    // Only clean "+ -" and "- -" when followed by a digit, letter, or √/^ symbol,
    // NOT when followed by ( which indicates a parenthesized expression like (-1)²
    if let Ok(re_plus_minus) = Regex::new(r"\+ -([0-9a-zA-Z√^])") {
        result = re_plus_minus.replace_all(&result, "- $1").to_string();
    }

    if let Ok(re_minus_minus) = Regex::new(r"- -([0-9a-zA-Z√^])") {
        result = re_minus_minus.replace_all(&result, "+ $1").to_string();
    }

    // Also handle without space variants, but only before digits/letters
    if let Ok(re_plus_minus_compact) = Regex::new(r"\+-([0-9a-zA-Z])") {
        result = re_plus_minus_compact
            .replace_all(&result, "-$1")
            .to_string();
    }

    if let Ok(re_minus_minus_compact) = Regex::new(r"--([0-9a-zA-Z])") {
        result = re_minus_minus_compact
            .replace_all(&result, "+$1")
            .to_string();
    }

    result
}

/// Display an expression, automatically rendering poly_result as formatted polynomial.
/// This is the preferred way to display expressions that might be poly_result.
fn display_expr_or_poly(ctx: &Context, id: ExprId) -> String {
    // Try to render as poly_result first (fast path for opaque polynomials)
    if let Some(poly_str) = cas_engine::poly_store::try_render_poly_result(ctx, id) {
        return poly_str;
    }

    // Fall back to standard display
    clean_display_string(&format!("{}", DisplayExpr { context: ctx, id }))
}

/// Render an expression with scoped display transforms based on the rule name.
/// Used for per-step rendering where certain rules (e.g., "Quadratic Formula")
/// should display sqrt notation instead of ^(1/2).
/// Now also respects global style preferences for consistent root display.
fn render_with_rule_scope(
    ctx: &Context,
    id: ExprId,
    rule_name: &str,
    style_prefs: &StylePreferences,
) -> String {
    // Map rule names to scopes
    let scopes: Vec<cas_ast::display_transforms::ScopeTag> = match rule_name {
        "Quadratic Formula" => vec![cas_ast::display_transforms::ScopeTag::Rule(
            "QuadraticFormula",
        )],
        // Add more rule mappings as needed
        _ => vec![],
    };

    if scopes.is_empty() {
        // No transforms apply - use styled display with preferences
        DisplayExprStyled::new(ctx, id, style_prefs).to_string()
    } else {
        // Use scoped renderer with transforms
        // Note: ScopedRenderer has its own display logic, so we can't easily combine with style_prefs
        // For now, use the scoped renderer directly; future improvement could merge these systems
        let registry = cas_ast::display_transforms::DisplayTransformRegistry::with_defaults();
        let renderer = cas_ast::display_transforms::ScopedRenderer::new(ctx, &scopes, &registry);
        renderer.render(id)
    }
}

pub struct Repl {
    /// Core logic - pure computation without I/O
    pub core: ReplCore,
    /// Output verbosity level (UI concern)
    verbosity: Verbosity,
    /// CLI configuration (loaded from file, applies rules to core)
    config: CasConfig,
}

/// Substitute occurrences of `target` with `replacement` anywhere in the expression tree.
/// This is more robust than path-based reconstruction because it finds by identity, not position.
#[allow(dead_code)]
fn substitute_expr_by_id(
    context: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
) -> ExprId {
    // If this node is the target, return replacement
    if root == target {
        return replacement;
    }
    match context.get(root).clone() {
        Expr::Add(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l == l && new_r == r {
                root
            } else {
                context.add(Expr::Add(new_l, new_r))
            }
        }
        Expr::Mul(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l == l && new_r == r {
                root
            } else {
                context.add(Expr::Mul(new_l, new_r))
            }
        }
        Expr::Div(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l == l && new_r == r {
                root
            } else {
                context.add(Expr::Div(new_l, new_r))
            }
        }
        Expr::Pow(base, exp) => {
            let new_base = substitute_expr_by_id(context, base, target, replacement);
            let new_exp = substitute_expr_by_id(context, exp, target, replacement);
            if new_base == base && new_exp == exp {
                root
            } else {
                context.add(Expr::Pow(new_base, new_exp))
            }
        }
        Expr::Neg(e) => {
            let new_e = substitute_expr_by_id(context, e, target, replacement);
            if new_e == e {
                root
            } else {
                context.add(Expr::Neg(new_e))
            }
        }
        Expr::Function(name, args) => {
            let new_args: Vec<_> = args
                .iter()
                .map(|&a| substitute_expr_by_id(context, a, target, replacement))
                .collect();
            if new_args == args {
                root
            } else {
                context.add(Expr::Function(name, new_args))
            }
        }
        _ => root,
    }
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
        // Special case: Sub(a,b) may have been canonicalized to Add(a, Neg(b))
        // When PathStep::Right expects to modify the original "b", we need to
        // traverse into the Neg wrapper and reconstruct there.
        (Expr::Add(l, r), PathStep::Right) => {
            // Check if right side is Neg - if so, this might be a canonicalized Sub
            if let Expr::Neg(inner) = context.get(r).clone() {
                // Traverse into the Neg and wrap result back in Neg
                let new_inner =
                    reconstruct_global_expr(context, inner, remaining_path, replacement);
                let new_neg = context.add(Expr::Neg(new_inner));
                context.add(Expr::Add(l, new_neg))
            } else {
                // Normal case - not a canonicalized Sub
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
                root // Should not happen if path is valid
            }
        }
        _ => root, // Path mismatch or invalid structure
    }
}

fn should_show_step(step: &cas_engine::step::Step, verbosity: Verbosity) -> bool {
    use cas_engine::step::ImportanceLevel;

    match verbosity {
        Verbosity::None => false,
        Verbosity::Verbose => true,
        // Succinct and Normal both show Medium+ importance, just different display
        Verbosity::Succinct | Verbosity::Normal => {
            if step.get_importance() < ImportanceLevel::Medium {
                return false;
            }

            // Additional check: global no-ops (expansion then contraction cycles)
            if let (Some(before), Some(after)) = (step.global_before, step.global_after) {
                if before == after {
                    return false;
                }
            }

            true
        }
    }
}

impl Default for Repl {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Repl implementation - split across modules for maintainability
// =============================================================================
// We intentionally keep Repl's implementation split by concern, but *without*
// include!(), so the compiler can track boundaries and tooling/navigation works
// better. Each file defines `impl Repl { ... }` for its feature area.
//
// File contents:
//   init.rs             - Constructor and configuration sync
//   dispatch.rs         - Command dispatch and routing
//   help.rs             - Help system and documentation
//   commands_misc.rs    - Miscellaneous commands (set, show, debug, etc.)
//   semantics.rs        - Semantic analysis commands
//   commands_algebra.rs - Algebra commands (factor, expand, etc.)
//   commands_solve.rs   - Solve command and equation handling
//   show_steps.rs       - Step-by-step output formatting
//   eval.rs             - Expression evaluation
//   simplify.rs         - Simplification pipeline
//   rationalize.rs      - Rationalization commands
//   limit.rs            - Limit computation
//   free_fns.rs         - Free functions (format helpers, etc.)
// =============================================================================

mod commands_algebra;
mod commands_misc;
mod commands_solve;
mod commands_system;
mod core;
mod dispatch;
mod error_render;
mod eval;
mod free_fns;
mod help;
mod init;
mod limit;
pub mod output;
mod rationalize;
mod semantics;
mod show_steps;
mod simplify;
pub mod wire;

#[cfg(test)]
mod core_tests;

// Re-export core types for external use
pub use core::ReplCore;
pub use output::{reply_output, CoreResult, ReplMsg, ReplReply, ReplReplyExt, UiDelta};

// These were historically plain module-level helpers (when using include!()).
// Re-export them into this module scope so existing code can keep calling them
// directly (e.g. `display_solution_set(...)`) without `free_fns::` prefixes.
#[allow(unused_imports)]
use free_fns::*;
