//! Expand strategy selector for choosing optimal expansion method.
//!
//! Selects between Normal, MultinomialExact, and MultiPolyModpSafe strategies
//! based on expression complexity and coefficient bounds.

use cas_ast::{Context, Expr, ExprId};
use num_traits::ToPrimitive;

/// Features extracted from expression for strategy selection
#[derive(Debug, Clone)]
pub struct ExpandFeatures {
    /// Total node count
    pub nodes: usize,
    /// General polynomial-like structure (Add/Sub/Mul/Pow/Neg/Const/Sym)
    pub poly_like: bool,
    /// Strict eligibility for MultiPoly (no sqrt, trig, div, etc.)
    pub multipoly_eligible: bool,
    /// Count of Pow(Add(...), n) patterns
    pub pow_add_count: usize,
    /// Maximum exponent in Pow patterns
    pub max_pow_n: u32,
    /// Estimated output terms (capped at 1M)
    pub est_terms: u64,
    /// Composite log2 bound of maximum coefficient magnitude
    pub log2_coeff_bound: f64,
}

/// Expansion strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandStrategy {
    /// Normal symbolic expansion (distribute/rewrites)
    Normal,
    /// Fast multinomial expansion (exact integers)
    MultinomialExact,
    /// MultiPoly mod-p with coefficient guard
    MultiPolyModpSafe,
}

/// Policy for strategy selection
#[derive(Debug, Clone)]
pub struct ExpandPolicy {
    /// Minimum nodes to consider fast path (default: 200)
    pub min_nodes_for_fast: usize,
    /// Minimum estimated terms for mod-p (default: 2000)
    pub min_est_terms_for_modp: u64,
    /// Maximum log2(coeff) for safe mod-p reconstruction
    /// Should be log2(p) - 9 for 9-bit margin (~43 for p≈2^52)
    pub max_log2_coeff_for_modp: f64,
    /// Enable two-prime cross-check for extra safety (default: false)
    pub enable_two_prime_check: bool,
}

impl Default for ExpandPolicy {
    fn default() -> Self {
        Self {
            min_nodes_for_fast: 200,
            min_est_terms_for_modp: 2000,
            // DEFAULT_PRIME ≈ 4.5e15 ≈ 2^52, so log2(p/2) ≈ 51
            // With 9-bit margin: 51 - 9 = 42
            max_log2_coeff_for_modp: 42.0,
            enable_two_prime_check: false,
        }
    }
}

/// Extract features from expression for strategy selection
pub fn extract_expand_features(ctx: &Context, expr: ExprId) -> ExpandFeatures {
    let mut features = ExpandFeatures {
        nodes: 0,
        poly_like: true,
        multipoly_eligible: true,
        pow_add_count: 0,
        max_pow_n: 0,
        est_terms: 1,
        log2_coeff_bound: 0.0,
    };

    extract_features_recursive(ctx, expr, &mut features);
    features
}

fn extract_features_recursive(ctx: &Context, expr: ExprId, features: &mut ExpandFeatures) {
    features.nodes += 1;

    match ctx.get(expr) {
        Expr::Add(l, r) | Expr::Sub(l, r) => {
            extract_features_recursive(ctx, *l, features);
            extract_features_recursive(ctx, *r, features);
        }
        Expr::Mul(l, r) => {
            extract_features_recursive(ctx, *l, features);
            extract_features_recursive(ctx, *r, features);
        }
        Expr::Neg(e) => {
            extract_features_recursive(ctx, *e, features);
        }
        Expr::Pow(base, exp) => {
            // Check if exponent is positive integer
            if let Some(n) = extract_positive_int(ctx, *exp) {
                features.max_pow_n = features.max_pow_n.max(n);

                // Check if base is Add (multinomial pattern)
                if is_add_like(ctx, *base) {
                    features.pow_add_count += 1;

                    // Estimate terms: C(n+k-1, k-1) where k = base term count
                    let k = count_add_terms(ctx, *base) as u64;
                    let est = estimate_multinomial_terms(n as u64, k);
                    features.est_terms = features.est_terms.saturating_mul(est).min(1_000_000);

                    // Compute coefficient bound: n * log2(sum_abs_coeffs)
                    let s = sum_abs_coeffs(ctx, *base);
                    if s > 0.0 {
                        let bound = (n as f64) * s.log2();
                        features.log2_coeff_bound = features.log2_coeff_bound.max(bound);
                    }
                }

                extract_features_recursive(ctx, *base, features);
            } else {
                // Non-integer or negative exponent → not multipoly eligible
                features.multipoly_eligible = false;
                features.poly_like = false;
                extract_features_recursive(ctx, *base, features);
            }
        }
        Expr::Div(_, denom) => {
            // Division makes it not multipoly eligible (unless denom is const)
            if !is_constant(ctx, *denom) {
                features.multipoly_eligible = false;
            }
            features.poly_like = false;
        }
        Expr::Function(name, _) => {
            // Functions (sqrt, sin, ln, etc.) → not polynomial-like
            if name != "expand" {
                features.poly_like = false;
                features.multipoly_eligible = false;
            }
        }
        Expr::Number(_) | Expr::Variable(_) | Expr::SessionRef(_) => {
            // Atoms are fine
        }
        // Constant and Matrix are not polynomial-like
        Expr::Constant(_) | Expr::Matrix { .. } => {
            features.poly_like = false;
            features.multipoly_eligible = false;
        }
    }
}

/// Choose expansion strategy based on features
pub fn choose_expand_strategy(features: &ExpandFeatures, policy: &ExpandPolicy) -> ExpandStrategy {
    // Small expressions → try MultinomialExact (fast and exact)
    if features.nodes < policy.min_nodes_for_fast {
        return ExpandStrategy::MultinomialExact;
    }

    // If not eligible for multipoly → MultinomialExact or Normal
    if !features.multipoly_eligible {
        if features.pow_add_count > 0 {
            return ExpandStrategy::MultinomialExact;
        }
        return ExpandStrategy::Normal;
    }

    // Large expression that IS multipoly eligible
    // Check if mod-p is safe (coefficient bound check)
    if features.est_terms >= policy.min_est_terms_for_modp
        && features.log2_coeff_bound < policy.max_log2_coeff_for_modp
    {
        return ExpandStrategy::MultiPolyModpSafe;
    }

    // Default: try MultinomialExact first
    ExpandStrategy::MultinomialExact
}

/// Check if coefficient bound is safe for mod-p reconstruction
#[inline]
pub fn coeff_bound_is_safe(log2_bound: f64, p: u64) -> bool {
    let log2_p = (p as f64).log2();
    let max_safe = log2_p - 1.0 - 8.0; // 8-bit margin
    log2_bound < max_safe
}

// =============================================================================
// Helper functions
// =============================================================================

fn extract_positive_int(ctx: &Context, expr: ExprId) -> Option<u32> {
    if let Expr::Number(n) = ctx.get(expr) {
        // Check if non-negative integer
        if n.is_integer() && n >= &num_rational::BigRational::from_integer(0.into()) {
            return n.to_integer().to_u32();
        }
    }
    None
}

fn is_add_like(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
}

fn is_constant(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(_))
}

fn count_add_terms(ctx: &Context, expr: ExprId) -> usize {
    match ctx.get(expr) {
        Expr::Add(l, r) | Expr::Sub(l, r) => count_add_terms(ctx, *l) + count_add_terms(ctx, *r),
        Expr::Neg(e) => count_add_terms(ctx, *e),
        _ => 1,
    }
}

/// Sum of absolute values of coefficients in an Add expression
fn sum_abs_coeffs(ctx: &Context, expr: ExprId) -> f64 {
    match ctx.get(expr) {
        Expr::Add(l, r) | Expr::Sub(l, r) => sum_abs_coeffs(ctx, *l) + sum_abs_coeffs(ctx, *r),
        Expr::Neg(e) => sum_abs_coeffs(ctx, *e),
        Expr::Mul(l, r) => {
            // Try to extract coefficient
            let cl = extract_coeff_value(ctx, *l);
            let cr = extract_coeff_value(ctx, *r);
            if let Some(c) = cl {
                c.abs() * sum_abs_coeffs(ctx, *r).max(1.0)
            } else if let Some(c) = cr {
                c.abs() * sum_abs_coeffs(ctx, *l).max(1.0)
            } else {
                1.0 // Unknown coefficient, assume 1
            }
        }
        Expr::Number(n) => n.to_f64().unwrap_or(1.0).abs(),
        _ => 1.0, // Variables count as coefficient 1
    }
}

fn extract_coeff_value(ctx: &Context, expr: ExprId) -> Option<f64> {
    if let Expr::Number(n) = ctx.get(expr) {
        return n.to_f64();
    }
    None
}

/// Estimate multinomial terms: C(n+k-1, k-1) capped at 1M
fn estimate_multinomial_terms(n: u64, k: u64) -> u64 {
    if k <= 1 {
        return 1;
    }
    // C(n+k-1, k-1) = (n+k-1)! / ((n)! * (k-1)!)
    // Use approximate calculation to avoid overflow
    let mut result: u64 = 1;
    for i in 0..(k - 1) {
        result = result.saturating_mul(n + k - 1 - i);
        result /= i + 1;
        if result > 1_000_000 {
            return 1_000_000;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_policy() {
        let policy = ExpandPolicy::default();
        assert_eq!(policy.min_nodes_for_fast, 200);
        assert_eq!(policy.min_est_terms_for_modp, 2000);
    }

    #[test]
    fn test_estimate_multinomial_terms() {
        // C(7+8-1, 8-1) = C(14, 7) = 3432
        assert_eq!(estimate_multinomial_terms(7, 8), 3432);
        // C(2+3-1, 3-1) = C(4, 2) = 6
        assert_eq!(estimate_multinomial_terms(2, 3), 6);
    }
}
