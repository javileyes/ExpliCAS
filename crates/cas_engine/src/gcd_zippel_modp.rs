//! Zippel modular GCD algorithm for multivariate polynomials.
//!
//! This implements a recursive Zippel-style GCD over Fp:
//! 1. Base case: 1 variable → Euclidean GCD
//! 2. Recursive: evaluate one variable at multiple points,
//!    compute GCD recursively on each, then interpolate
//!
//! Avoids tensor-grid explosion by processing one variable at a time.

use crate::modp::{add_mod, inv_mod, sub_mod};
use crate::mono::Mono;
use crate::multipoly_modp::MultiPolyModP;
use crate::unipoly_modp::UniPolyModP;

/// Budget for Zippel GCD algorithm
#[derive(Clone, Debug)]
pub struct ZippelBudget {
    /// Maximum interpolation points per variable (deg+1 is minimum needed)
    pub max_points_per_var: usize,
    /// Maximum retries for bad evaluation points
    pub max_retries: usize,
    /// Number of random trials for probabilistic verification
    pub verify_trials: usize,
    /// Enable debug output
    pub debug: bool,
}

impl Default for ZippelBudget {
    fn default() -> Self {
        Self {
            max_points_per_var: 16,
            max_retries: 8,
            verify_trials: 6,
            debug: false,
        }
    }
}

/// Tuned budget for mm_gcd benchmark (7 variables, degree 7)
pub fn budget_for_mm_gcd() -> ZippelBudget {
    ZippelBudget {
        max_points_per_var: 9, // deg 7 needs 8 points, +1 margin
        max_retries: 64,       // Many retries instead of many points
        verify_trials: 10,     // Thorough verification
        debug: true,           // See what's happening
    }
}

/// Seed points for evaluation - extended list avoiding 0 and 1
const SEED_POINTS: [u64; 48] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223,
];

// =============================================================================
// Main entry points
// =============================================================================

/// Compute GCD of two multivariate polynomials mod p using Zippel algorithm.
/// Returns monic GCD (leading coefficient = 1).
pub fn gcd_zippel_modp(
    p: &MultiPolyModP,
    q: &MultiPolyModP,
    budget: &ZippelBudget,
) -> Option<MultiPolyModP> {
    gcd_zippel_modp_impl(p, q, None, budget)
}

/// Compute GCD with forced main variable (useful for benchmarks).
pub fn gcd_zippel_modp_with_main(
    p: &MultiPolyModP,
    q: &MultiPolyModP,
    main_var: usize,
    budget: &ZippelBudget,
) -> Option<MultiPolyModP> {
    gcd_zippel_modp_impl(p, q, Some(main_var), budget)
}

fn gcd_zippel_modp_impl(
    p: &MultiPolyModP,
    q: &MultiPolyModP,
    forced_main: Option<usize>,
    budget: &ZippelBudget,
) -> Option<MultiPolyModP> {
    // Sanity checks
    if p.p != q.p {
        if budget.debug {
            eprintln!("[Zippel] ERROR: prime mismatch");
        }
        return None;
    }
    if p.num_vars != q.num_vars {
        if budget.debug {
            eprintln!("[Zippel] ERROR: num_vars mismatch");
        }
        return None;
    }
    if p.is_zero() {
        let mut result = q.clone();
        result.make_monic();
        return Some(result);
    }
    if q.is_zero() {
        let mut result = p.clone();
        result.make_monic();
        return Some(result);
    }

    let num_vars = p.num_vars;
    if num_vars == 0 {
        return Some(MultiPolyModP::constant(1, p.p, 0));
    }

    // Build list of active variables
    let active_vars: Vec<usize> = (0..num_vars)
        .filter(|&i| p.degree_in(i) > 0 || q.degree_in(i) > 0)
        .collect();

    if active_vars.is_empty() {
        return Some(MultiPolyModP::constant(1, p.p, num_vars));
    }

    if budget.debug {
        eprintln!(
            "[Zippel] Starting: {} active vars, p={} terms, q={} terms",
            active_vars.len(),
            p.num_terms(),
            q.num_terms()
        );
    }

    // Recursive GCD with optional forced main var
    let mut result = gcd_zippel_rec(p, q, &active_vars, forced_main, 0, budget)?;
    result.make_monic();

    // Probabilistic verification
    if budget.debug {
        eprintln!(
            "[Zippel] Verifying result: {} terms, degree {}",
            result.num_terms(),
            result.total_degree()
        );
    }

    if !verify_by_univar_evals(&result, p, q, budget) {
        if budget.debug {
            eprintln!("[Zippel] FAILED: verification");
        }
        return None;
    }

    if budget.debug {
        eprintln!("[Zippel] SUCCESS");
    }
    Some(result)
}

// =============================================================================
// Recursive Zippel core with instrumentation
// =============================================================================

fn gcd_zippel_rec(
    p: &MultiPolyModP,
    q: &MultiPolyModP,
    active_vars: &[usize],
    forced_main: Option<usize>,
    depth: usize,
    budget: &ZippelBudget,
) -> Option<MultiPolyModP> {
    let p_mod = p.p;
    let num_vars = p.num_vars;
    let indent = "  ".repeat(depth);

    // Base case: single variable → use univariate GCD
    if active_vars.len() == 1 {
        let var = active_vars[0];
        let p_uni = poly_to_univar(p, var);
        let q_uni = poly_to_univar(q, var);
        let mut g_uni = p_uni.gcd(&q_uni);
        g_uni.make_monic(); // Ensure monic for consistency

        if budget.debug {
            eprintln!(
                "{}[depth={}] Base case var={}: gcd degree={}",
                indent,
                depth,
                var,
                g_uni.degree()
            );
        }
        return Some(univar_to_multipoly(&g_uni, var, num_vars, p_mod));
    }

    // Choose variable to evaluate
    let eval_var = if let Some(main) = forced_main {
        if active_vars.contains(&main) {
            main
        } else {
            choose_eval_var(p, q, active_vars)
        }
    } else {
        choose_eval_var(p, q, active_vars)
    };

    // Reduced active vars (without eval_var)
    let reduced_vars: Vec<usize> = active_vars
        .iter()
        .copied()
        .filter(|&v| v != eval_var)
        .collect();

    // Degree bounds
    let deg_p = p.degree_in(eval_var) as usize;
    let deg_q = q.degree_in(eval_var) as usize;
    let deg_bound = deg_p.min(deg_q);
    let num_points = (deg_bound + 1).min(budget.max_points_per_var);

    if budget.debug {
        eprintln!(
            "{}[depth={}] Eval var={}, deg_p={}, deg_q={}, need {} points",
            indent, depth, eval_var, deg_p, deg_q, num_points
        );
    }

    // Collect samples with degree-stability filter
    let mut samples: Vec<(u64, MultiPolyModP)> = Vec::new();
    let mut expected_gcd_deg: Option<u32> = None;
    let mut seed_idx = 0;
    let mut retries = 0;
    let mut skipped_degree = 0;
    let mut skipped_zero = 0;

    while samples.len() < num_points && retries < budget.max_retries {
        if seed_idx >= SEED_POINTS.len() {
            seed_idx = 0;
            retries += 1;
        }

        let t = SEED_POINTS[seed_idx] % p_mod;
        seed_idx += 1;

        // Evaluate both polynomials at eval_var = t
        let p_t = eval_var_at(p, eval_var, t);
        let q_t = eval_var_at(q, eval_var, t);

        // Skip if evaluation gave zero (unlucky point)
        if p_t.is_zero() || q_t.is_zero() {
            skipped_zero += 1;
            continue;
        }

        // Degree-stability filter: check that degrees didn't drop unexpectedly
        // This catches unlucky evaluation points
        if reduced_vars.len() == 1 {
            let rv = reduced_vars[0];
            let p_t_deg = p_t.degree_in(rv);
            let q_t_deg = q_t.degree_in(rv);
            let expected_p_deg = p.degree_in(rv);
            let expected_q_deg = q.degree_in(rv);

            if p_t_deg < expected_p_deg || q_t_deg < expected_q_deg {
                skipped_degree += 1;
                continue;
            }
        }

        // Recursive GCD on reduced polynomial (no forced main for inner calls)
        let mut g_t = gcd_zippel_rec(&p_t, &q_t, &reduced_vars, None, depth + 1, budget)?;
        g_t.make_monic(); // Ensure monic for consistent interpolation

        let this_deg = g_t.total_degree();

        // Check degree consistency
        match expected_gcd_deg {
            None => {
                expected_gcd_deg = Some(this_deg);
            }
            Some(exp_deg) => {
                if this_deg > exp_deg {
                    // Higher degree - unlucky point, skip
                    skipped_degree += 1;
                    continue;
                }
                if this_deg < exp_deg {
                    // Lower degree - previous samples were unlucky, restart
                    if budget.debug {
                        eprintln!(
                            "{}  Restarting: got deg {} < expected {}",
                            indent, this_deg, exp_deg
                        );
                    }
                    samples.clear();
                    expected_gcd_deg = Some(this_deg);
                }
            }
        }

        samples.push((t, g_t));
    }

    if budget.debug {
        eprintln!(
            "{}[depth={}] Collected {} samples (skipped: {} zero, {} degree)",
            indent,
            depth,
            samples.len(),
            skipped_zero,
            skipped_degree
        );
    }

    if samples.len() < num_points {
        if budget.debug {
            eprintln!(
                "{}[depth={}] FAILED: not enough samples ({} < {})",
                indent,
                depth,
                samples.len(),
                num_points
            );
        }
        return None;
    }

    // Interpolate in eval_var
    let result = lagrange_interpolate_poly(eval_var, &samples, p_mod, num_vars);

    if budget.debug && result.is_some() {
        let r = result.as_ref().unwrap();
        eprintln!(
            "{}[depth={}] Interpolated: {} terms, degree {}",
            indent,
            depth,
            r.num_terms(),
            r.total_degree()
        );
    }

    result
}

// =============================================================================
// Variable selection
// =============================================================================

/// Choose which variable to evaluate next.
/// Strategy: pick the one with smallest min(deg_p, deg_q) to minimize interpolation points.
fn choose_eval_var(p: &MultiPolyModP, q: &MultiPolyModP, active_vars: &[usize]) -> usize {
    active_vars
        .iter()
        .copied()
        .min_by_key(|&v| p.degree_in(v).min(q.degree_in(v)))
        .unwrap_or(active_vars[0])
}

// =============================================================================
// Evaluation helpers
// =============================================================================

fn eval_var_at(poly: &MultiPolyModP, var_idx: usize, val: u64) -> MultiPolyModP {
    poly.eval_vars(&[(var_idx, val)])
}

fn poly_to_univar(poly: &MultiPolyModP, var_idx: usize) -> UniPolyModP {
    let max_deg = poly.degree_in(var_idx) as usize;
    let mut coeffs = vec![0u64; max_deg + 1];

    for (mono, coeff) in &poly.terms {
        let deg = mono.deg_var(var_idx) as usize;
        coeffs[deg] = add_mod(coeffs[deg], *coeff, poly.p);
    }

    let mut result = UniPolyModP { p: poly.p, coeffs };
    result.trim();
    result
}

fn univar_to_multipoly(
    uni: &UniPolyModP,
    var_idx: usize,
    num_vars: usize,
    p_mod: u64,
) -> MultiPolyModP {
    let mut terms = Vec::new();

    for (deg, &coeff) in uni.coeffs.iter().enumerate() {
        if coeff != 0 {
            let mut mono = Mono::zero();
            mono.0[var_idx] = deg as u16;
            terms.push((mono, coeff));
        }
    }

    terms.sort_by(|a, b| b.0.cmp(&a.0));

    MultiPolyModP {
        p: p_mod,
        num_vars,
        terms,
    }
}

// =============================================================================
// Lagrange interpolation
// =============================================================================

fn lagrange_interpolate_poly(
    eval_var: usize,
    samples: &[(u64, MultiPolyModP)],
    p_mod: u64,
    num_vars: usize,
) -> Option<MultiPolyModP> {
    if samples.is_empty() {
        return None;
    }
    if samples.len() == 1 {
        return Some(samples[0].1.clone());
    }

    let k = samples.len();
    let points: Vec<u64> = samples.iter().map(|(t, _)| *t).collect();

    let mut result = MultiPolyModP::zero(p_mod, num_vars);

    for j in 0..k {
        let (t_j, g_j) = &samples[j];
        let l_j = lagrange_basis_poly(eval_var, *t_j, &points, p_mod, num_vars)?;
        let term = g_j.mul(&l_j);
        result = result.add(&term);
    }

    result.normalize();
    Some(result)
}

fn lagrange_basis_poly(
    var_idx: usize,
    t_j: u64,
    all_points: &[u64],
    p_mod: u64,
    num_vars: usize,
) -> Option<MultiPolyModP> {
    let mut result = MultiPolyModP::constant(1, p_mod, num_vars);
    let x_poly = MultiPolyModP::var(var_idx, p_mod, num_vars);

    for &t_m in all_points {
        if t_m == t_j {
            continue;
        }

        let diff = sub_mod(t_j, t_m, p_mod);
        let diff_inv = inv_mod(diff, p_mod)?;

        let t_m_const = MultiPolyModP::constant(t_m, p_mod, num_vars);
        let factor = x_poly.sub(&t_m_const);

        result = result.mul(&factor);
        result = result.mul_scalar(diff_inv);
    }

    Some(result)
}

// =============================================================================
// Probabilistic verification
// =============================================================================

fn verify_by_univar_evals(
    g: &MultiPolyModP,
    p: &MultiPolyModP,
    q: &MultiPolyModP,
    budget: &ZippelBudget,
) -> bool {
    if g.is_zero() {
        return false;
    }
    if g.num_vars == 0 || (g.terms.len() == 1 && g.terms[0].0.is_constant()) {
        return true;
    }

    let test_var = (0..g.num_vars).find(|&i| g.degree_in(i) > 0).unwrap_or(0);
    let p_mod = g.p;

    for trial in 0..budget.verify_trials {
        let mut assignments: Vec<(usize, u64)> = Vec::new();
        for v in 0..g.num_vars {
            if v != test_var {
                let seed_idx = (trial * g.num_vars + v) % SEED_POINTS.len();
                assignments.push((v, SEED_POINTS[seed_idx] % p_mod));
            }
        }

        let g_uni = g.eval_to_univar(test_var, &assignments);
        let p_uni = p.eval_to_univar(test_var, &assignments);
        let q_uni = q.eval_to_univar(test_var, &assignments);

        if g_uni.is_zero() {
            continue;
        }

        if let Some((_, rem_p)) = p_uni.div_rem(&g_uni) {
            if !rem_p.is_zero() {
                return false;
            }
        } else {
            return false;
        }

        if let Some((_, rem_q)) = q_uni.div_rem(&g_uni) {
            if !rem_q.is_zero() {
                return false;
            }
        } else {
            return false;
        }
    }

    true
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 1_000_000_007;

    #[test]
    fn test_gcd_univar() {
        let x = MultiPolyModP::var(0, P, 1);
        let one = MultiPolyModP::constant(1, P, 1);
        let p = x.pow(2).sub(&one);
        let q = x.sub(&one);

        let budget = ZippelBudget::default();
        let g = gcd_zippel_modp(&p, &q, &budget).unwrap();

        assert_eq!(g.total_degree(), 1);
        assert_eq!(g.degree_in(0), 1);
    }

    #[test]
    fn test_gcd_bivar() {
        let x = MultiPolyModP::var(0, P, 2);
        let y = MultiPolyModP::var(1, P, 2);
        let one = MultiPolyModP::constant(1, P, 2);

        let xy = x.add(&y);
        let xp1 = x.add(&one);
        let yp1 = y.add(&one);

        let p = xy.mul(&xp1);
        let q = xy.mul(&yp1);

        let budget = ZippelBudget::default();
        let g = gcd_zippel_modp(&p, &q, &budget).unwrap();

        assert_eq!(g.total_degree(), 1);
    }

    #[test]
    fn test_gcd_trivar() {
        let x = MultiPolyModP::var(0, P, 3);
        let y = MultiPolyModP::var(1, P, 3);
        let z = MultiPolyModP::var(2, P, 3);
        let one = MultiPolyModP::constant(1, P, 3);

        let xyz = x.add(&y).add(&z);
        let xp1 = x.add(&one);
        let zp1 = z.add(&one);

        let p = xyz.mul(&xp1);
        let q = xyz.mul(&zp1);

        let budget = ZippelBudget::default();
        let g = gcd_zippel_modp(&p, &q, &budget).unwrap();

        assert_eq!(g.total_degree(), 1);
    }

    #[test]
    fn test_univar_conversion() {
        let x = MultiPolyModP::var(0, P, 2);
        let one = MultiPolyModP::constant(1, P, 2);
        let xp1 = x.add(&one);

        let uni = xp1.eval_to_univar(0, &[(1, 5)]);
        assert_eq!(uni.degree(), 1);
        assert_eq!(uni.coeffs[0], 1);
        assert_eq!(uni.coeffs[1], 1);
    }

    #[test]
    fn test_with_forced_main() {
        let x = MultiPolyModP::var(0, P, 2);
        let y = MultiPolyModP::var(1, P, 2);
        let one = MultiPolyModP::constant(1, P, 2);

        let xy = x.add(&y);
        let xp1 = x.add(&one);
        let yp1 = y.add(&one);

        let p = xy.mul(&xp1);
        let q = xy.mul(&yp1);

        let budget = ZippelBudget::default();

        // Force main to be y (var 1)
        let g = gcd_zippel_modp_with_main(&p, &q, 1, &budget).unwrap();
        assert_eq!(g.total_degree(), 1);
    }
}
