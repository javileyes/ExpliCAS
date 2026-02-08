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

use crate::step::Step;
use cas_ast::{Context, Expr, ExprId};
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
    /// Expression before the operation (plain text or LaTeX depending on context)
    pub before_expr: String,
    /// Expression after the operation (plain text or LaTeX depending on context)
    pub after_expr: String,
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
    use fraction_steps::FractionSumInfo;
    use nested_fractions::{
        classify_nested_fraction, contains_div, extract_combined_fraction_str,
        NestedFractionPattern,
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
        assert!(!contains_div(&ctx, x));

        // 1/x contains div
        let div = ctx.add(Expr::Div(one, x));
        assert!(contains_div(&ctx, div));

        // 1 + 1/x contains div
        let add = ctx.add(Expr::Add(one, div));
        assert!(contains_div(&ctx, add));
    }
}
