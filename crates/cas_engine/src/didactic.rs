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

use crate::step::Step;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

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
    /// LaTeX representation of expression before
    pub before_latex: String,
    /// LaTeX representation of expression after
    pub after_latex: String,
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
        if step.description.starts_with("Simplified fraction by GCD") {
            sub_steps.extend(generate_gcd_factorization_substeps(ctx, step));
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

/// Find all fraction sums in an expression tree
fn find_all_fraction_sums(ctx: &Context, expr: ExprId) -> Vec<FractionSumInfo> {
    let mut results = Vec::new();
    find_all_fraction_sums_recursive(ctx, expr, &mut results);
    results
}

fn find_all_fraction_sums_recursive(
    ctx: &Context,
    expr: ExprId,
    results: &mut Vec<FractionSumInfo>,
) {
    // Check if this is an Add chain of fractions
    if let Some(info) = find_fraction_sum_in_expr(ctx, expr) {
        results.push(info);
    }

    // Recurse into children
    match ctx.get(expr) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            find_all_fraction_sums_recursive(ctx, *l, results);
            find_all_fraction_sums_recursive(ctx, *r, results);
        }
        Expr::Pow(b, e) => {
            find_all_fraction_sums_recursive(ctx, *b, results);
            find_all_fraction_sums_recursive(ctx, *e, results);
        }
        Expr::Neg(e) => find_all_fraction_sums_recursive(ctx, *e, results),
        Expr::Function(_, args) => {
            for arg in args {
                find_all_fraction_sums_recursive(ctx, *arg, results);
            }
        }
        _ => {}
    }
}

/// Information about a fraction sum that was computed
#[derive(Debug)]
struct FractionSumInfo {
    /// The fractions that were summed
    fractions: Vec<BigRational>,
    /// The result of the sum
    result: BigRational,
}

/// Detect if between this step and the previous one, an exponent changed
/// due to fraction arithmetic
fn detect_exponent_fraction_change(
    ctx: &Context,
    steps: &[Step],
    step_idx: usize,
) -> Option<FractionSumInfo> {
    let current_step = &steps[step_idx];

    // We need to check if there's a fraction sum in the GLOBAL expression
    // that will be simplified silently (not its own step)

    // For "Add Inverse" rule - this is when x^(a) - x^(a) = 0, meaning
    // the fractional exponent sum was already computed silently
    if current_step.rule_name.contains("Inverse") || current_step.rule_name.contains("Power") {
        // Check the global_after of previous step if available,
        // or use global_after of current step
        let global_expr = if step_idx > 0 {
            steps[step_idx - 1]
                .global_after
                .unwrap_or(steps[step_idx - 1].after)
        } else {
            current_step.global_after.unwrap_or(current_step.before)
        };

        // Search recursively for fraction sums
        if let Some(info) = find_fraction_sum_in_expr(ctx, global_expr) {
            return Some(info);
        }
    }

    None
}

/// Search an expression tree for Add chains of fractions
fn find_fraction_sum_in_expr(ctx: &Context, expr: ExprId) -> Option<FractionSumInfo> {
    match ctx.get(expr) {
        Expr::Add(_, _) => {
            // Collect all terms in the Add chain
            let mut terms = Vec::new();
            collect_add_terms(ctx, expr, &mut terms);

            // Check if all terms are fractions (Number or Div(Number,Number))
            let mut fractions = Vec::new();
            for term in &terms {
                if let Some(frac) = try_as_fraction(ctx, *term) {
                    fractions.push(frac);
                } else {
                    return None; // Not all terms are fractions
                }
            }

            if fractions.len() >= 2 {
                // Only consider it a fraction sum if at least one fraction has denominator != 1
                // This filters out integer sums like 1 + 4 which aren't really "fraction sums"
                let has_actual_fraction = fractions.iter().any(|f| !f.denom().is_one());
                if !has_actual_fraction {
                    return None;
                }
                let result: BigRational = fractions.iter().cloned().sum();
                return Some(FractionSumInfo { fractions, result });
            }
            None
        }
        Expr::Pow(_, e) => {
            // Check the exponent for fraction sums
            find_fraction_sum_in_expr(ctx, *e)
        }
        Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Sub(l, r) => {
            // Check both sides
            find_fraction_sum_in_expr(ctx, *l).or_else(|| find_fraction_sum_in_expr(ctx, *r))
        }
        Expr::Neg(e) => find_fraction_sum_in_expr(ctx, *e),
        Expr::Function(_, args) => {
            for arg in args {
                if let Some(info) = find_fraction_sum_in_expr(ctx, *arg) {
                    return Some(info);
                }
            }
            None
        }
        _ => None,
    }
}

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

/// Generate sub-steps explaining how fractions were summed
fn generate_fraction_sum_substeps(info: &FractionSumInfo) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if info.fractions.len() < 2 {
        return sub_steps;
    }

    // Step 1: Show the original sum
    let original_sum: Vec<String> = info.fractions.iter().map(format_fraction).collect();

    // Step 2: Find common denominator
    let lcm = info
        .fractions
        .iter()
        .fold(BigInt::from(1), |acc, f| lcm_bigint(&acc, f.denom()));

    // Step 3: Show conversion to common denominator
    let converted: Vec<String> = info
        .fractions
        .iter()
        .map(|f| {
            let multiplier = &lcm / f.denom();
            let new_numer = f.numer() * &multiplier;
            format!("\\frac{{{}}}{{{}}}", new_numer, lcm)
        })
        .collect();

    // Only add sub-steps if there's actual conversion needed
    let needs_conversion = info.fractions.iter().any(|f| f.denom() != &lcm);

    if needs_conversion {
        sub_steps.push(SubStep {
            description: format!("Find common denominator: {}", lcm),
            before_latex: original_sum.join(" + "),
            after_latex: converted.join(" + "),
        });
    }

    // Step 4: Show the result
    sub_steps.push(SubStep {
        description: "Sum the fractions".to_string(),
        before_latex: if needs_conversion {
            converted.join(" + ")
        } else {
            original_sum.join(" + ")
        },
        after_latex: format_fraction(&info.result),
    });

    sub_steps
}

/// Generate sub-steps explaining polynomial factorization and GCD cancellation
/// For example: (x² - 4) / (2 + x) shows:
///   1. Factor numerator: x² - 4 → (x-2)(x+2)
///   2. Cancel common factor: (x-2)(x+2) / (x+2) → x-2
fn generate_gcd_factorization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    use cas_ast::DisplayExpr;

    let mut sub_steps = Vec::new();

    // Extract GCD from description: "Simplified fraction by GCD: <gcd>"
    let gcd_start = "Simplified fraction by GCD: ";
    if !step.description.starts_with(gcd_start) {
        return sub_steps;
    }
    let gcd_str = &step.description[gcd_start.len()..];

    // Get the before expression (which should be a Div)
    let before_expr = step.before;
    if let Expr::Div(num, den) = ctx.get(before_expr) {
        let num_str = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: *num
            }
        );
        let den_str = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: *den
            }
        );
        let after_str = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: step.after
            }
        );

        // Check if the step's "before_local" shows factored form in Rule display
        // If available, use it to show the factorization step
        if let Some(local_before) = step.before_local {
            let local_before_str = format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: local_before
                }
            );

            // Sub-step 1: Factor the expression
            // If the rule display shows factored form like "(x - 2) * (2 + x) / (2 + x)"
            // we can extract the factorization
            if local_before_str.contains(&gcd_str) && local_before_str.contains('*') {
                sub_steps.push(SubStep {
                    description: format!("Factor: {} contains factor {}", num_str, gcd_str),
                    before_latex: num_str.clone(),
                    after_latex: local_before_str
                        .split('/')
                        .next()
                        .unwrap_or(&local_before_str)
                        .trim()
                        .to_string(),
                });
            }
        }

        // Sub-step 2: Cancel common factor
        // Wrap in parentheses if they contain operators
        let needs_parens_num =
            num_str.contains('+') || num_str.contains('-') || num_str.contains(' ');
        let needs_parens_den =
            den_str.contains('+') || den_str.contains('-') || den_str.contains(' ');
        let before_formatted = format!(
            "{} / {}",
            if needs_parens_num {
                format!("({})", num_str)
            } else {
                num_str.clone()
            },
            if needs_parens_den {
                format!("({})", den_str)
            } else {
                den_str.clone()
            }
        );
        sub_steps.push(SubStep {
            description: format!("Cancel common factor: {}", gcd_str),
            before_latex: before_formatted,
            after_latex: after_str,
        });
    }

    sub_steps
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
}
