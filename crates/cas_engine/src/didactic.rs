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
use cas_ast::{Context, Expr, ExprId, LaTeXExpr};
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
pub fn enrich_steps(ctx: &Context, _original_expr: ExprId, steps: Vec<Step>) -> Vec<EnrichedStep> {
    let mut enriched = Vec::with_capacity(steps.len());

    for (i, step) in steps.iter().enumerate() {
        let mut sub_steps = Vec::new();

        // Check for fraction sum in exponent
        if let Some(fraction_info) = detect_exponent_fraction_change(ctx, &steps, i) {
            sub_steps.extend(generate_fraction_sum_substeps(&fraction_info));
        }

        enriched.push(EnrichedStep {
            base_step: step.clone(),
            sub_steps,
        });
    }

    enriched
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
    if step_idx == 0 {
        return None;
    }

    let current_step = &steps[step_idx];
    let prev_step = &steps[step_idx - 1];

    // Look for changes in exponent expressions
    // Compare global_after of previous step with before of current step
    let prev_global = prev_step.global_after.unwrap_or(prev_step.after);
    let curr_before = current_step.before;

    // Check if this step involves a power rule and the exponent simplified
    if current_step.rule_name.contains("Power") || current_step.rule_name.contains("Inverse") {
        // Look for fraction sums in the expression
        if let Some(info) = find_fraction_sum_in_expr(ctx, curr_before) {
            return Some(info);
        }
    }

    // Also check the after expression for contrast
    let _curr_after = current_step.after;

    None
}

/// Search an expression tree for Add chains of fractions
fn find_fraction_sum_in_expr(ctx: &Context, expr: ExprId) -> Option<FractionSumInfo> {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            // Collect all terms in the Add chain
            let mut terms = Vec::new();
            collect_add_terms(ctx, expr, &mut terms);

            // Check if all terms are numbers (fractions)
            let mut fractions = Vec::new();
            for term in &terms {
                if let Expr::Number(n) = ctx.get(*term) {
                    fractions.push(n.clone());
                } else {
                    return None; // Not all terms are numbers
                }
            }

            if fractions.len() >= 2 {
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
    let original_sum: Vec<String> = info.fractions.iter().map(|f| format_fraction(f)).collect();

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
