//! Zippel modular GCD algorithm for multivariate polynomials.
//!
//! This implements a recursive Zippel-style GCD over Fp:
//! 1. Base case: 1 variable → Euclidean GCD
//! 2. Recursive: evaluate one variable at multiple points,
//!    compute GCD recursively on each, then interpolate
//!
//! Avoids tensor-grid explosion by processing one variable at a time.
//!
//! Optimization: Uses Rayon for parallel point evaluation at top levels.

use crate::modp::{add_mod, inv_mod, sub_mod};
use crate::mono::Mono;
use crate::multipoly_modp::MultiPolyModP;
use crate::unipoly_modp::UniPolyModP;

#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;

/// Budget for Zippel GCD algorithm
#[derive(Clone, Debug)]
pub struct ZippelBudget {
    /// Maximum interpolation points per variable (deg+1 is minimum needed)
    pub max_points_per_var: usize,
    /// Maximum retries for bad evaluation points
    pub max_retries: usize,
    /// Number of random trials for probabilistic verification
    pub verify_trials: usize,
    /// Force a specific main variable (if Some), bypassing auto-selection
    pub forced_main_var: Option<usize>,
}

/// Preset configurations for Zippel GCD
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZippelPreset {
    /// Conservative settings: more points, more verification (reliable)
    Safe,
    /// Balanced settings: fewer points, less verification (faster)
    Aggressive,
    /// Tuned for mm_gcd benchmark: minimal points for deg 7, fast verify
    MmGcd,
}

impl ZippelPreset {
    /// Parse preset from string (case-insensitive)
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "safe" => Some(Self::Safe),
            "aggressive" | "fast" => Some(Self::Aggressive),
            "mm_gcd" | "mmgcd" | "mm" => Some(Self::MmGcd),
            _ => None,
        }
    }
}

impl ZippelBudget {
    /// Create budget from preset
    pub fn for_preset(preset: ZippelPreset) -> Self {
        match preset {
            ZippelPreset::Safe => Self {
                max_points_per_var: 16,
                max_retries: 32,
                verify_trials: 6,
                forced_main_var: None,
            },
            ZippelPreset::Aggressive => Self {
                max_points_per_var: 10,
                max_retries: 16,
                verify_trials: 4,
                forced_main_var: None,
            },
            ZippelPreset::MmGcd => Self {
                max_points_per_var: 8,
                max_retries: 8,
                verify_trials: 3,
                forced_main_var: None,
            },
        }
    }

    /// Apply forced main variable to this budget
    pub fn with_main_var(mut self, main_var: Option<usize>) -> Self {
        self.forced_main_var = main_var;
        self
    }
}

/// Check if debug tracing is enabled (via CAS_ZIPPEL_TRACE env var)
#[inline]
fn is_trace_enabled() -> bool {
    std::env::var("CAS_ZIPPEL_TRACE").is_ok()
}

impl Default for ZippelBudget {
    fn default() -> Self {
        Self::for_preset(ZippelPreset::Safe)
    }
}

/// Tuned budget for mm_gcd benchmark (7 variables, degree 7)
/// Prefer using ZippelBudget::for_preset(ZippelPreset::MmGcd) instead.
pub fn budget_for_mm_gcd() -> ZippelBudget {
    ZippelBudget::for_preset(ZippelPreset::MmGcd)
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
/// Uses budget.forced_main_var if set, otherwise auto-selects main variable.
pub fn gcd_zippel_modp(
    p: &MultiPolyModP,
    q: &MultiPolyModP,
    budget: &ZippelBudget,
) -> Option<MultiPolyModP> {
    gcd_zippel_modp_impl(p, q, budget.forced_main_var, budget)
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
        if is_trace_enabled() {
            eprintln!("[Zippel] ERROR: prime mismatch");
        }
        return None;
    }
    if p.num_vars != q.num_vars {
        if is_trace_enabled() {
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

    if is_trace_enabled() {
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
    if is_trace_enabled() {
        eprintln!(
            "[Zippel] Verifying result: {} terms, degree {}",
            result.num_terms(),
            result.total_degree()
        );
    }

    if !verify_by_univar_evals(&result, p, q, budget) {
        if is_trace_enabled() {
            eprintln!("[Zippel] FAILED: verification");
        }
        return None;
    }

    if is_trace_enabled() {
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

        if is_trace_enabled() {
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
            choose_eval_var(p, q, active_vars, budget, depth)
        }
    } else {
        choose_eval_var(p, q, active_vars, budget, depth)
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

    if is_trace_enabled() {
        eprintln!(
            "{}[depth={}] Eval var={}, deg_p={}, deg_q={}, need {} points",
            indent, depth, eval_var, deg_p, deg_q, num_points
        );
    }

    // Collect samples - use parallel evaluation at depth 0 for large polys
    let use_parallel = should_use_parallel(depth, p.num_terms(), q.num_terms());

    if is_trace_enabled() {
        let compiled_par = cfg!(feature = "parallel");
        #[cfg(feature = "parallel")]
        let threads = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let threads = 1_usize;
        eprintln!(
            "{}[depth={}] use_parallel={}, compiled_par={}, threads={}, p_terms={}, q_terms={}",
            indent,
            depth,
            use_parallel,
            compiled_par,
            threads,
            p.num_terms(),
            q.num_terms()
        );
    }

    let samples = if use_parallel {
        collect_samples_parallel(
            p,
            q,
            eval_var,
            &reduced_vars,
            num_points,
            budget,
            p_mod,
            depth,
        )?
    } else {
        collect_samples_sequential(
            p,
            q,
            eval_var,
            &reduced_vars,
            num_points,
            budget,
            p_mod,
            depth,
        )?
    };

    let skipped_zero = 0; // For trace compatibility
    let skipped_degree = 0;

    if is_trace_enabled() {
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
        if is_trace_enabled() {
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

    if is_trace_enabled() {
        if let Some(ref r) = result {
            eprintln!(
                "{}[depth={}] Interpolated: {} terms, degree {}",
                indent,
                depth,
                r.num_terms(),
                r.total_degree()
            );
        }
    }

    result
}

// =============================================================================
// Parallel sampling configuration
// =============================================================================

/// Threshold for parallel evaluation
const PAR_DEPTH: usize = 2; // Parallelize at depth 0, 1, 2
const PAR_TERM_THRESHOLD: usize = 5_000; // Minimum combined terms for parallelism

/// Decide whether to use parallel evaluation
fn should_use_parallel(depth: usize, p_terms: usize, q_terms: usize) -> bool {
    #[cfg(feature = "parallel")]
    {
        depth <= PAR_DEPTH && (p_terms + q_terms) >= PAR_TERM_THRESHOLD
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (depth, p_terms, q_terms);
        false
    }
}

/// Sample result for parallel collection
#[allow(dead_code)]
struct Sample {
    point: u64,
    gcd: MultiPolyModP,
    degree: u32,
}

/// Collect samples using parallel evaluation (deterministic ordering)
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)] // GCD algorithm requires all these distinct parameters
fn collect_samples_parallel(
    p: &MultiPolyModP,
    q: &MultiPolyModP,
    eval_var: usize,
    reduced_vars: &[usize],
    num_points: usize,
    budget: &ZippelBudget,
    p_mod: u64,
    depth: usize,
) -> Option<Vec<(u64, MultiPolyModP)>> {
    use rayon::prelude::*;

    // Prepare batch of points to evaluate
    let points: Vec<u64> = SEED_POINTS[..num_points.min(SEED_POINTS.len())]
        .iter()
        .map(|&pt| pt % p_mod)
        .collect();

    if is_trace_enabled() {
        eprintln!(
            "  [depth={}] Parallel eval of {} points",
            depth,
            points.len()
        );
    }

    // Parallel evaluation: each thread does eval + recursive gcd
    let results: Vec<Option<Sample>> = points
        .par_iter()
        .map(|&t| {
            // Evaluate both polynomials at eval_var = t
            let p_t = eval_var_at(p, eval_var, t);
            let q_t = eval_var_at(q, eval_var, t);

            // Skip if evaluation gave zero
            if p_t.is_zero() || q_t.is_zero() {
                return None;
            }

            // Degree-stability filter
            if reduced_vars.len() == 1 {
                let rv = reduced_vars[0];
                if p_t.degree_in(rv) < p.degree_in(rv) || q_t.degree_in(rv) < q.degree_in(rv) {
                    return None;
                }
            }

            // Recursive GCD (no forced main for inner, sequential from here)
            let mut g_t = gcd_zippel_rec(&p_t, &q_t, reduced_vars, None, depth + 1, budget)?;
            g_t.make_monic();

            let degree = g_t.total_degree();
            Some(Sample {
                point: t,
                gcd: g_t,
                degree,
            })
        })
        .collect();

    // Process results in deterministic order: take first N valid with consistent degree
    let mut samples: Vec<(u64, MultiPolyModP)> = Vec::with_capacity(num_points);
    let mut expected_deg: Option<u32> = None;

    for (pt, maybe_sample) in points.into_iter().zip(results.into_iter()) {
        if let Some(sample) = maybe_sample {
            match expected_deg {
                None => {
                    expected_deg = Some(sample.degree);
                    samples.push((pt, sample.gcd));
                }
                Some(exp) => {
                    if sample.degree == exp {
                        samples.push((pt, sample.gcd));
                    } else if sample.degree < exp {
                        // Lower degree is better - restart with this
                        samples.clear();
                        expected_deg = Some(sample.degree);
                        samples.push((pt, sample.gcd));
                    }
                    // Higher degree: skip
                }
            }
        }
        if samples.len() >= num_points {
            break;
        }
    }

    if samples.len() >= num_points {
        Some(samples)
    } else {
        None
    }
}

/// Sequential fallback for collect_samples (used when parallel disabled or small polys)
#[allow(clippy::too_many_arguments)] // GCD algorithm requires all these distinct parameters
fn collect_samples_sequential(
    p: &MultiPolyModP,
    q: &MultiPolyModP,
    eval_var: usize,
    reduced_vars: &[usize],
    num_points: usize,
    budget: &ZippelBudget,
    p_mod: u64,
    depth: usize,
) -> Option<Vec<(u64, MultiPolyModP)>> {
    let mut samples: Vec<(u64, MultiPolyModP)> = Vec::new();
    let mut expected_gcd_deg: Option<u32> = None;
    let mut seed_idx = 0;
    let mut retries = 0;

    while samples.len() < num_points && retries < budget.max_retries {
        if seed_idx >= SEED_POINTS.len() {
            seed_idx = 0;
            retries += 1;
        }

        let t = SEED_POINTS[seed_idx] % p_mod;
        seed_idx += 1;

        let p_t = eval_var_at(p, eval_var, t);
        let q_t = eval_var_at(q, eval_var, t);

        if p_t.is_zero() || q_t.is_zero() {
            continue;
        }

        if reduced_vars.len() == 1 {
            let rv = reduced_vars[0];
            if p_t.degree_in(rv) < p.degree_in(rv) || q_t.degree_in(rv) < q.degree_in(rv) {
                continue;
            }
        }

        let mut g_t = gcd_zippel_rec(&p_t, &q_t, reduced_vars, None, depth + 1, budget)?;
        g_t.make_monic();

        let this_deg = g_t.total_degree();

        match expected_gcd_deg {
            None => {
                expected_gcd_deg = Some(this_deg);
            }
            Some(exp_deg) => {
                if this_deg > exp_deg {
                    continue;
                }
                if this_deg < exp_deg {
                    samples.clear();
                    expected_gcd_deg = Some(this_deg);
                }
            }
        }

        samples.push((t, g_t));
    }

    if samples.len() >= num_points {
        Some(samples)
    } else {
        None
    }
}

/// Fallback for non-parallel feature
#[cfg(not(feature = "parallel"))]
fn collect_samples_parallel(
    p: &MultiPolyModP,
    q: &MultiPolyModP,
    eval_var: usize,
    reduced_vars: &[usize],
    num_points: usize,
    budget: &ZippelBudget,
    p_mod: u64,
    depth: usize,
) -> Option<Vec<(u64, MultiPolyModP)>> {
    collect_samples_sequential(
        p,
        q,
        eval_var,
        reduced_vars,
        num_points,
        budget,
        p_mod,
        depth,
    )
}

// =============================================================================
// Variable selection with smart scoring
// =============================================================================

/// Metrics for evaluating variable selection quality
#[derive(Debug, Clone)]
struct VarMetrics {
    v: usize,
    _deg_min: u32,     // min(deg_p, deg_q) in v - reserved for future heuristics
    points: u32,       // deg_min + 1 (interpolation points needed)
    lead_support: u32, // min(leading terms in p, leading terms in q)
    bucket_cost: u64,  // sum of squared bucket sizes (approximates coef complexity)
}

/// Compute bucket cost for a polynomial on variable v.
/// This measures how "expensive" it will be to work with univar coefficients.
/// Lower is better: sparse univariates have lower bucket_cost.
fn bucket_cost(poly: &MultiPolyModP, v: usize) -> u64 {
    let deg = poly.degree_in(v) as usize;
    if deg == 0 {
        return 1; // Constant in this variable
    }

    // Count terms per degree bucket
    let mut counts = vec![0u32; deg + 1];
    for (mono, _) in &poly.terms {
        let d = mono.deg_var(v) as usize;
        if d <= deg {
            counts[d] += 1;
        }
    }

    // Sum of squared counts (penalizes dense buckets)
    counts.iter().map(|&c| (c as u64) * (c as u64)).sum()
}

/// Count terms at leading degree for variable v.
/// Higher is better: more leading support means less likely degree drop.
fn leading_support(poly: &MultiPolyModP, v: usize) -> u32 {
    let lead_deg = poly.degree_in(v);
    if lead_deg == 0 {
        return poly.num_terms() as u32;
    }

    poly.terms
        .iter()
        .filter(|(mono, _)| mono.deg_var(v) == lead_deg)
        .count() as u32
}

/// Choose which variable to evaluate next using smart scoring.
///
/// Scoring criteria (in order of priority):
/// 1. Minimize points needed (deg_min + 1)
/// 2. Maximize leading support (reduce degree drop risk)
/// 3. Minimize bucket cost (simpler univar coefficients)
/// 4. Tie-break by variable index (determinism)
fn choose_eval_var(
    p: &MultiPolyModP,
    q: &MultiPolyModP,
    active_vars: &[usize],
    budget: &ZippelBudget,
    depth: usize,
) -> usize {
    use std::cmp::Reverse;

    let metrics: Vec<VarMetrics> = active_vars
        .iter()
        .copied()
        .map(|v| {
            let deg_p = p.degree_in(v) as u32;
            let deg_q = q.degree_in(v) as u32;
            let deg_min = deg_p.min(deg_q);
            let points = (deg_min + 1).min(budget.max_points_per_var as u32);
            let lead_sup = leading_support(p, v).min(leading_support(q, v));
            let bc = bucket_cost(p, v) + bucket_cost(q, v);

            VarMetrics {
                v,
                _deg_min: deg_min,
                points,
                lead_support: lead_sup,
                bucket_cost: bc,
            }
        })
        .collect();

    // Log metrics at depth 0 only (for debugging)
    if is_trace_enabled() && depth == 0 && !metrics.is_empty() {
        eprintln!("[Zippel] Variable scoring (depth={}):", depth);
        for m in &metrics {
            eprintln!(
                "  var={}: points={}, lead_support={}, bucket_cost={}",
                m.v, m.points, m.lead_support, m.bucket_cost
            );
        }
    }

    // Sort by: (points ASC, lead_support DESC, bucket_cost ASC, v ASC)
    let best = metrics
        .iter()
        .min_by_key(|m| (m.points, Reverse(m.lead_support), m.bucket_cost, m.v))
        .map(|m| m.v)
        .unwrap_or(active_vars[0]);

    if is_trace_enabled() && depth == 0 {
        eprintln!("[Zippel] Chose main_var={}", best);
    }

    best
}

// =============================================================================
// Evaluation helpers
// =============================================================================

fn eval_var_at(poly: &MultiPolyModP, var_idx: usize, val: u64) -> MultiPolyModP {
    poly.eval_var_fast(var_idx, val)
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

    let points: Vec<u64> = samples.iter().map(|(t, _)| *t).collect();

    let mut result = MultiPolyModP::zero(p_mod, num_vars);

    for (t_j, g_j) in samples.iter() {
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
