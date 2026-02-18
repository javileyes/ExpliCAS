//! GCD computation layers for MultiPoly (Layer 2 and Layer 2.5).

use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::BTreeMap;

use cas_math::polynomial::Polynomial;

use super::arithmetic::gcd_bigrational;
use super::{Monomial, MultiPoly, PolyBudget, VarIdx};

// =============================================================================
// Layer 2: Evaluation, Interpolation, Heuristic GCD
// =============================================================================

impl MultiPoly {
    /// Evaluate a single variable to a rational value
    /// Returns a new polynomial with one fewer variable
    pub fn eval_var(&self, var_idx: VarIdx, val: &BigRational) -> Self {
        if var_idx >= self.vars.len() {
            return self.clone();
        }

        let mut new_vars = self.vars.clone();
        new_vars.remove(var_idx);

        let mut map: BTreeMap<Monomial, BigRational> = BTreeMap::new();

        for (coeff, mono) in &self.terms {
            let exp = mono[var_idx];
            // Compute val^exp
            let mut val_pow = BigRational::one();
            for _ in 0..exp {
                val_pow *= val;
            }
            // New coefficient = old_coeff * val^exp
            let new_coeff = coeff * &val_pow;

            // New monomial without this variable
            let mut new_mono = mono.clone();
            new_mono.remove(var_idx);

            let entry = map.entry(new_mono).or_insert_with(BigRational::zero);
            *entry = entry.clone() + new_coeff;
        }

        Self::from_map(new_vars, map)
    }

    /// Evaluate all variables except main_var, returns univariate Polynomial
    pub fn eval_to_univariate(
        &self,
        main_var: VarIdx,
        assigns: &[(VarIdx, BigRational)],
    ) -> Option<Polynomial> {
        if main_var >= self.vars.len() {
            return None;
        }

        // Apply all assignments
        let mut p = self.clone();
        // Sort assigns in reverse order so indices don't shift
        let mut sorted_assigns: Vec<_> = assigns.to_vec();
        sorted_assigns.sort_by(|a, b| b.0.cmp(&a.0));

        for (var, val) in sorted_assigns {
            if var != main_var && var < p.vars.len() {
                p = p.eval_var(var, &val);
            }
        }

        // Now p should have only main_var (at new index 0 since others removed)
        if p.vars.len() != 1 {
            return None;
        }

        // Convert to Polynomial
        let var_name = p.vars[0].clone();
        let max_deg = p.degree_in(0);
        let mut coeffs = vec![BigRational::zero(); (max_deg + 1) as usize];

        for (coeff, mono) in &p.terms {
            let exp = mono[0] as usize;
            if exp < coeffs.len() {
                coeffs[exp] = coeff.clone();
            }
        }

        Some(Polynomial::new(coeffs, var_name))
    }

    /// Evaluate a variable and adjust remaining indices for main/param vars
    pub(crate) fn eval_var_adjusted(
        &self,
        var_idx: VarIdx,
        val: &BigRational,
        main: VarIdx,
        param: VarIdx,
    ) -> Self {
        if var_idx >= self.vars.len() || var_idx == main || var_idx == param {
            return self.clone();
        }

        let mut new_vars = self.vars.clone();
        new_vars.remove(var_idx);

        let mut map: BTreeMap<Monomial, BigRational> = BTreeMap::new();

        for (coeff, mono) in &self.terms {
            let exp = mono[var_idx];
            let mut val_pow = BigRational::one();
            for _ in 0..exp {
                val_pow *= val;
            }
            let new_coeff = coeff * &val_pow;

            let mut new_mono = mono.clone();
            new_mono.remove(var_idx);

            let entry = map.entry(new_mono).or_insert_with(BigRational::zero);
            *entry = entry.clone() + new_coeff;
        }

        Self::from_map(new_vars, map)
    }
}

/// Budget for heuristic GCD computation
#[derive(Clone, Debug)]
pub struct GcdBudget {
    pub max_points: usize,
    pub max_total_terms: usize,
    pub max_seeds: usize,
}

impl Default for GcdBudget {
    fn default() -> Self {
        Self {
            max_points: 8,
            max_total_terms: 200,
            max_seeds: 4,
        }
    }
}

/// Enum indicating which GCD technique was used
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GcdLayer {
    /// Monomial and content GCD (fastest)
    Layer1MonomialContent,
    /// Heuristic bivariate with seed assignments
    Layer2HeuristicSeeds,
    /// Tensor grid interpolation for multi-param factors
    Layer25TensorGrid,
}

/// Compute scaled GCD of two univariate polynomials
/// Returns gcd * gcd(lc(p), lc(q)) to preserve parameter-dependent factors
fn scaled_gcd_sample(p: &Polynomial, q: &Polynomial) -> Option<Polynomial> {
    if p.is_zero() {
        return Some(q.clone());
    }
    if q.is_zero() {
        return Some(p.clone());
    }

    // Compute monic GCD
    let g = p.gcd(q);
    if g.is_zero() {
        return Some(g);
    }

    // Scale by gcd of leading coefficients
    let lc_p = p.leading_coeff();
    let lc_q = q.leading_coeff();
    let lc_gcd = gcd_bigrational(&lc_p, &lc_q);

    // Multiply g by lc_gcd
    let scale = Polynomial::new(vec![lc_gcd], g.var.clone());
    Some(g.mul(&scale))
}

/// Lagrange interpolation: given points (x_i, y_i), find polynomial P with P(x_i) = y_i
fn interpolate_lagrange(points: &[(BigRational, BigRational)], var: &str) -> Option<Polynomial> {
    if points.is_empty() {
        return None;
    }
    if points.len() == 1 {
        // Constant polynomial
        return Some(Polynomial::new(vec![points[0].1.clone()], var.to_string()));
    }

    let mut result = Polynomial::zero(var.to_string());

    for (i, (xi, yi)) in points.iter().enumerate() {
        // Build Lagrange basis polynomial L_i(x) = prod_{j!=i} (x - x_j) / (x_i - x_j)
        let mut basis = Polynomial::one(var.to_string());

        for (j, (xj, _)) in points.iter().enumerate() {
            if i == j {
                continue;
            }

            // (x - x_j)
            let linear = Polynomial::new(vec![-xj.clone(), BigRational::one()], var.to_string());
            basis = basis.mul(&linear);

            // Divide by (x_i - x_j)
            let denom = xi - xj;
            if denom.is_zero() {
                return None; // Duplicate x values
            }
            let inv_denom = BigRational::one() / denom;
            let scale = Polynomial::new(vec![inv_denom], var.to_string());
            basis = basis.mul(&scale);
        }

        // Add y_i * L_i(x)
        let scale = Polynomial::new(vec![yi.clone()], var.to_string());
        let term = basis.mul(&scale);
        result = result.add(&term);
    }

    Some(result)
}

/// Heuristic multivariate GCD using evaluation and interpolation
/// Supports N variables (N >= 2) by fixing extra vars to seed values
pub fn gcd_multivar_layer2(p: &MultiPoly, q: &MultiPoly, budget: &GcdBudget) -> Option<MultiPoly> {
    if p.vars.len() < 2 {
        return None;
    }
    if p.vars != q.vars {
        return None;
    }
    if p.is_zero() || q.is_zero() {
        return None;
    }
    if p.num_terms() + q.num_terms() > budget.max_total_terms {
        return None; // Too complex
    }

    let n_vars = p.vars.len();
    let vars = &p.vars;

    // Seed values for fixing extra variables: 2, 3, 5, 7, -1, -2, ...
    let seed_values: Vec<Vec<BigRational>> = generate_seed_combinations(n_vars, budget.max_seeds);

    // Choose main/param variable candidates based on min degree
    let var_candidates = choose_var_candidates(p, q);

    for (main_var, param_var) in &var_candidates {
        let main_var = *main_var;
        let param_var = *param_var;

        // Expected degrees
        let deg_p = p.degree_in(main_var);
        let deg_q = q.degree_in(main_var);
        let expected_gcd_deg = deg_p.min(deg_q);

        if expected_gcd_deg == 0 {
            continue;
        }

        let param_deg_p = p.degree_in(param_var);
        let param_deg_q = q.degree_in(param_var);
        let _max_coeff_deg = param_deg_p.min(param_deg_q);

        // Try different seeds for the extra variables
        for seed in &seed_values {
            // Build fixed assignments for all vars except main and param
            let fixed_assigns: Vec<(VarIdx, BigRational)> = (0..n_vars)
                .filter(|&v| v != main_var && v != param_var)
                .enumerate()
                .map(|(i, v)| {
                    (
                        v,
                        seed.get(i)
                            .cloned()
                            .unwrap_or_else(|| BigRational::from_integer(2.into())),
                    )
                })
                .collect();

            // Reduce p and q by fixing extra vars
            let mut p_reduced = p.clone();
            let mut q_reduced = q.clone();
            for (var, val) in &fixed_assigns {
                // We need to remap indices after each eval_var
                // For simplicity, eval all at once using eval_multiple
                p_reduced = p_reduced.eval_var_adjusted(*var, val, main_var, param_var);
                q_reduced = q_reduced.eval_var_adjusted(*var, val, main_var, param_var);
            }

            // Now p_reduced and q_reduced should have only 2 vars (remapped to 0, 1)
            if p_reduced.vars.len() != 2 || q_reduced.vars.len() != 2 {
                continue;
            }

            // Run bivar interpolation
            if let Some(gcd_bivar) = try_bivar_layer2(&p_reduced, &q_reduced, budget) {
                // Lift candidate back to N-var by replacing variable names
                if let Some(candidate) = lift_to_nvar(&gcd_bivar, vars, main_var, param_var) {
                    // Verify on original polys
                    if p.div_exact(&candidate).is_some() && q.div_exact(&candidate).is_some() {
                        return Some(candidate);
                    }
                }
            }
        }
    }

    None
}

/// GCD with budget tracking, returning PassStats.
///
/// This is the instrumented version of `gcd_multivar_layer2` for unified budget charging.
pub fn gcd_multivar_layer2_with_stats(
    p: &MultiPoly,
    q: &MultiPoly,
    budget: &GcdBudget,
) -> (Option<MultiPoly>, crate::budget::PassStats) {
    let result = gcd_multivar_layer2(p, q, budget);

    let terms = result.as_ref().map_or(0, |g| g.num_terms() as u64);

    let stats = crate::budget::PassStats {
        op: crate::budget::Operation::PolyOps,
        rewrite_count: 0,
        nodes_delta: 0,
        terms_materialized: terms,
        poly_ops: 1, // One GCD operation
        stop_reason: None,
    };

    (result, stats)
}

/// Generate seed combinations for fixing extra variables
fn generate_seed_combinations(n_vars: usize, max_seeds: usize) -> Vec<Vec<BigRational>> {
    // For extra vars, we need n_vars - 2 values per seed
    let extra_count = n_vars.saturating_sub(2);
    if extra_count == 0 {
        return vec![vec![]];
    }

    // Simple seeds: small primes and negatives
    let base_vals: Vec<i64> = vec![2, 3, 5, 7, -1, -2, 11, 13];

    let mut seeds = Vec::new();
    for start in 0..max_seeds.min(base_vals.len()) {
        let seed: Vec<BigRational> = (0..extra_count)
            .map(|i| {
                let idx = (start + i) % base_vals.len();
                BigRational::from_integer(base_vals[idx].into())
            })
            .collect();
        seeds.push(seed);
    }

    if seeds.is_empty() {
        seeds.push(vec![BigRational::from_integer(2.into()); extra_count]);
    }

    seeds
}

/// Choose candidate (main_var, param_var) pairs based on degrees
fn choose_var_candidates(p: &MultiPoly, q: &MultiPoly) -> Vec<(VarIdx, VarIdx)> {
    let n = p.vars.len();
    let mut scored: Vec<(VarIdx, u32)> = (0..n)
        .map(|v| {
            let min_deg = p.degree_in(v).min(q.degree_in(v));
            (v, min_deg)
        })
        .collect();

    // Sort by min degree descending (higher = better main var)
    scored.sort_by(|a, b| b.1.cmp(&a.1));

    let mut candidates = Vec::new();
    // Try top 2-3 combinations
    for i in 0..scored.len().min(2) {
        for j in 0..scored.len().min(3) {
            if i != j {
                candidates.push((scored[i].0, scored[j].0));
            }
        }
    }

    if candidates.is_empty() && n >= 2 {
        candidates.push((0, 1));
    }

    candidates
}

/// Run bivar Layer 2 on a 2-variable polynomial pair
fn try_bivar_layer2(p: &MultiPoly, q: &MultiPoly, budget: &GcdBudget) -> Option<MultiPoly> {
    if p.vars.len() != 2 || q.vars.len() != 2 {
        return None;
    }

    let vars = &p.vars;

    for main_var in 0..2 {
        let param_var = 1 - main_var;

        let deg_p = p.degree_in(main_var);
        let deg_q = q.degree_in(main_var);
        let expected_gcd_deg = deg_p.min(deg_q);

        if expected_gcd_deg == 0 {
            continue;
        }

        let param_deg_p = p.degree_in(param_var);
        let param_deg_q = q.degree_in(param_var);
        let max_coeff_deg = param_deg_p.min(param_deg_q);

        let num_points = (max_coeff_deg as usize + 1).min(budget.max_points);
        let eval_points: Vec<BigRational> = generate_eval_points(num_points + 4); // Extra for bad points

        let mut samples: Vec<(BigRational, Polynomial)> = Vec::new();

        for point in &eval_points {
            let assigns = vec![(param_var, point.clone())];

            let p_uni = match p.eval_to_univariate(main_var, &assigns) {
                Some(u) => u,
                None => continue,
            };
            let q_uni = match q.eval_to_univariate(main_var, &assigns) {
                Some(u) => u,
                None => continue,
            };

            // Skip if degree dropped
            if p_uni.degree() < deg_p as usize || q_uni.degree() < deg_q as usize {
                continue;
            }

            let g = match scaled_gcd_sample(&p_uni, &q_uni) {
                Some(g) => g,
                None => continue,
            };

            if !samples.is_empty() {
                let first_deg = samples[0].1.degree();
                if g.degree() != first_deg {
                    break;
                }
            }

            samples.push((point.clone(), g));

            if samples.len() >= num_points {
                break;
            }
        }

        if samples.len() < (max_coeff_deg as usize + 1) {
            continue;
        }

        let gcd_deg = samples[0].1.degree();
        let mut gcd_coeffs: Vec<Polynomial> = Vec::new();

        for k in 0..=gcd_deg {
            let points: Vec<(BigRational, BigRational)> = samples
                .iter()
                .map(|(pt, poly)| {
                    let coeff = poly
                        .coeffs
                        .get(k)
                        .cloned()
                        .unwrap_or_else(BigRational::zero);
                    (pt.clone(), coeff)
                })
                .collect();

            let coeff_poly = interpolate_lagrange(&points, &vars[param_var])?;
            gcd_coeffs.push(coeff_poly);
        }

        let candidate = assemble_candidate(vars, main_var, param_var, &gcd_coeffs)?;

        // Verify
        if p.div_exact(&candidate).is_some() && q.div_exact(&candidate).is_some() {
            return Some(candidate);
        }
    }

    None
}

/// Lift a 2-var GCD back to N-var form
fn lift_to_nvar(
    gcd: &MultiPoly,
    original_vars: &[String],
    main: VarIdx,
    param: VarIdx,
) -> Option<MultiPoly> {
    if gcd.vars.len() != 2 {
        return None;
    }

    // Map gcd variables back to original positions
    let mut map: BTreeMap<Monomial, BigRational> = BTreeMap::new();
    let n = original_vars.len();

    for (coeff, mono) in &gcd.terms {
        let mut new_mono = vec![0u32; n];
        // mono[0] goes to main, mono[1] goes to param
        if !mono.is_empty() {
            new_mono[main] = mono[0];
        }
        if mono.len() >= 2 {
            new_mono[param] = mono[1];
        }

        let entry = map.entry(new_mono).or_insert_with(BigRational::zero);
        *entry = entry.clone() + coeff.clone();
    }

    Some(MultiPoly::from_map(original_vars.to_vec(), map))
}

/// Generate evaluation points: 0, 1, -1, 2, -2, 3, -3, ...
fn generate_eval_points(n: usize) -> Vec<BigRational> {
    let mut points = Vec::with_capacity(n);
    points.push(BigRational::zero());

    let mut k = 1i64;
    while points.len() < n {
        points.push(BigRational::from_integer(k.into()));
        if points.len() < n {
            points.push(BigRational::from_integer((-k).into()));
        }
        k += 1;
    }

    points
}

/// Assemble MultiPoly from interpolated coefficients
/// G(main, param) = sum_k coeff_k(param) * main^k
fn assemble_candidate(
    vars: &[String],
    main_var: VarIdx,
    param_var: VarIdx,
    coeffs_in_param: &[Polynomial],
) -> Option<MultiPoly> {
    if vars.len() != 2 {
        return None;
    }

    let mut map: BTreeMap<Monomial, BigRational> = BTreeMap::new();

    for (k, coeff_poly) in coeffs_in_param.iter().enumerate() {
        for (j, c) in coeff_poly.coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            // Monomial: main^k * param^j
            let mut mono = vec![0u32; 2];
            mono[main_var] = k as u32;
            mono[param_var] = j as u32;

            let entry = map.entry(mono).or_insert_with(BigRational::zero);
            *entry = entry.clone() + c.clone();
        }
    }

    Some(MultiPoly::from_map(vars.to_vec(), map))
}

// =============================================================================
// Layer 2.5: Zippel-lite Tensor Grid GCD
// =============================================================================

/// Budget for Layer 2.5 Zippel-lite GCD
#[derive(Clone, Debug)]
pub struct Layer25Budget {
    pub max_vars: usize,
    pub max_samples: usize,
    pub max_param_deg: u32,
    pub max_gcd_deg: usize,
}

impl Default for Layer25Budget {
    fn default() -> Self {
        Self {
            max_vars: 4,
            max_samples: 64,
            max_param_deg: 3,
            max_gcd_deg: 6,
        }
    }
}

/// Layer 2.5: Zippel-lite GCD using tensor grid interpolation
/// Handles GCD that depends on multiple parameter variables
pub fn gcd_multivar_layer25(
    p: &MultiPoly,
    q: &MultiPoly,
    budget: &Layer25Budget,
) -> Option<MultiPoly> {
    let n_vars = p.vars.len();

    // Pre-checks
    if n_vars < 2 || n_vars > budget.max_vars {
        return None;
    }
    if p.vars != q.vars {
        return None;
    }
    if p.is_zero() || q.is_zero() {
        return None;
    }

    let vars = &p.vars;

    // Choose main variable (highest min degree)
    let main_var = choose_main_var(p, q);
    let params: Vec<VarIdx> = (0..n_vars).filter(|&v| v != main_var).collect();

    if params.is_empty() {
        return None; // Only 1 var, use univar
    }

    // Expected GCD degree in main_var
    let deg_p = p.degree_in(main_var);
    let deg_q = q.degree_in(main_var);
    let max_gcd_deg = deg_p.min(deg_q) as usize;

    if max_gcd_deg == 0 || max_gcd_deg > budget.max_gcd_deg {
        return None;
    }

    // Compute degree bounds per param
    let param_degs: Vec<u32> = params
        .iter()
        .map(|&v| p.degree_in(v).min(q.degree_in(v)).min(budget.max_param_deg))
        .collect();

    // Build tensor grid
    let grid = build_tensor_grid(&params, &param_degs, budget.max_samples);
    if grid.is_empty() {
        return None;
    }

    // Sample univariate GCDs at each grid point
    let mut samples: Vec<(Vec<BigRational>, Polynomial)> = Vec::new();
    let mut target_deg: Option<usize> = None;

    for point in &grid {
        // Build assignment: vec of (var_idx, value)
        let assigns: Vec<(VarIdx, BigRational)> = params
            .iter()
            .zip(point.iter())
            .map(|(&v, val)| (v, val.clone()))
            .collect();

        // Evaluate to univariate
        let p_uni = match eval_all_but_main(p, main_var, &assigns) {
            Some(u) => u,
            None => continue,
        };
        let q_uni = match eval_all_but_main(q, main_var, &assigns) {
            Some(u) => u,
            None => continue,
        };

        // Skip if degree dropped
        if p_uni.degree() < deg_p as usize || q_uni.degree() < deg_q as usize {
            continue;
        }

        // Compute scaled GCD
        let g = match scaled_gcd_sample(&p_uni, &q_uni) {
            Some(g) => g,
            None => continue,
        };

        // Check degree consistency
        match target_deg {
            None => target_deg = Some(g.degree()),
            Some(td) => {
                if g.degree() != td {
                    continue; // Inconsistent, skip this point
                }
            }
        }

        samples.push((point.clone(), g));
    }

    let gcd_deg = target_deg?;
    if gcd_deg == 0 {
        return None; // Trivial GCD
    }

    // Need enough points for interpolation: product of (param_deg + 1)
    let min_points: usize = param_degs.iter().map(|&d| (d + 1) as usize).product();
    if samples.len() < min_points {
        return None;
    }

    // Interpolate each coefficient of GCD as MultiPoly in params
    let mut gcd_coeffs: Vec<MultiPoly> = Vec::new();
    let param_vars: Vec<String> = params.iter().map(|&v| vars[v].clone()).collect();

    for k in 0..=gcd_deg {
        // Extract coefficient k from each sample
        let coeff_samples: Vec<(Vec<BigRational>, BigRational)> = samples
            .iter()
            .map(|(pt, poly)| {
                let c = poly
                    .coeffs
                    .get(k)
                    .cloned()
                    .unwrap_or_else(BigRational::zero);
                (pt.clone(), c)
            })
            .collect();

        // Interpolate as MultiPoly in params
        let coeff_poly = interpolate_tensor(&param_vars, &param_degs, &coeff_samples)?;
        gcd_coeffs.push(coeff_poly);
    }

    // Build GCD: G = sum_k c_k(params) * main^k
    let candidate = build_gcd_multivar(vars, main_var, &params, &gcd_coeffs)?;

    // Verify by exact division
    if p.div_exact(&candidate).is_some() && q.div_exact(&candidate).is_some() {
        return Some(candidate);
    }

    None
}

/// Choose main variable based on max min-degree
fn choose_main_var(p: &MultiPoly, q: &MultiPoly) -> VarIdx {
    let n = p.vars.len();
    (0..n)
        .max_by_key(|&v| p.degree_in(v).min(q.degree_in(v)))
        .unwrap_or(0)
}

/// Build tensor grid of evaluation points
fn build_tensor_grid(
    params: &[VarIdx],
    param_degs: &[u32],
    max_samples: usize,
) -> Vec<Vec<BigRational>> {
    if params.is_empty() {
        return vec![vec![]];
    }

    // Points per param: degree + 1 for interpolation
    let points_per_param: Vec<Vec<BigRational>> = param_degs
        .iter()
        .map(|&deg| generate_eval_points((deg + 1) as usize))
        .collect();

    // Cartesian product
    let mut grid: Vec<Vec<BigRational>> = vec![vec![]];
    for pts in &points_per_param {
        let mut new_grid = Vec::new();
        for existing in &grid {
            for pt in pts {
                if new_grid.len() >= max_samples {
                    break;
                }
                let mut new_point = existing.clone();
                new_point.push(pt.clone());
                new_grid.push(new_point);
            }
            if new_grid.len() >= max_samples {
                break;
            }
        }
        grid = new_grid;
    }

    grid
}

/// Evaluate all variables except main to get univariate in main
fn eval_all_but_main(
    p: &MultiPoly,
    main_var: VarIdx,
    assigns: &[(VarIdx, BigRational)],
) -> Option<Polynomial> {
    p.eval_to_univariate(main_var, assigns)
}

/// Interpolate tensor grid samples into MultiPoly
fn interpolate_tensor(
    param_names: &[String],
    param_degs: &[u32],
    samples: &[(Vec<BigRational>, BigRational)],
) -> Option<MultiPoly> {
    if param_names.is_empty() {
        // Constant: should be single sample
        if samples.len() == 1 {
            return Some(MultiPoly::from_const(samples[0].1.clone()));
        }
        return None;
    }

    if param_names.len() == 1 {
        // Direct univariate Lagrange interpolation
        let points: Vec<(BigRational, BigRational)> = samples
            .iter()
            .map(|(pt, val)| (pt[0].clone(), val.clone()))
            .collect();
        let poly = interpolate_lagrange(&points, &param_names[0])?;

        // Convert Polynomial to MultiPoly
        return Some(polynomial_to_multipoly(&poly));
    }

    // Multi-variable: recursive interpolation
    // Group samples by first param value
    let first_var = &param_names[0];
    let first_deg = param_degs[0];
    let first_points = generate_eval_points((first_deg + 1) as usize);

    let rest_names: Vec<String> = param_names[1..].to_vec();
    let rest_degs: Vec<u32> = param_degs[1..].to_vec();

    // For each first_point, collect samples with that value and recurse
    let mut interpolated_coeffs: Vec<(BigRational, MultiPoly)> = Vec::new();

    for fp in &first_points {
        // Filter samples where first coord == fp
        let sub_samples: Vec<(Vec<BigRational>, BigRational)> = samples
            .iter()
            .filter(|(pt, _)| &pt[0] == fp)
            .map(|(pt, val)| (pt[1..].to_vec(), val.clone()))
            .collect();

        if sub_samples.is_empty() {
            continue;
        }

        // Recurse
        let sub_poly = interpolate_tensor(&rest_names, &rest_degs, &sub_samples)?;
        interpolated_coeffs.push((fp.clone(), sub_poly));
    }

    // Now interpolate in first_var using MultiPoly values
    lagrange_interpolate_multipoly(first_var, &interpolated_coeffs, &rest_names)
}

/// Convert Polynomial to MultiPoly (single variable)
fn polynomial_to_multipoly(poly: &Polynomial) -> MultiPoly {
    let mut map: BTreeMap<Monomial, BigRational> = BTreeMap::new();
    for (i, c) in poly.coeffs.iter().enumerate() {
        if !c.is_zero() {
            map.insert(vec![i as u32], c.clone());
        }
    }
    MultiPoly::from_map(vec![poly.var.clone()], map)
}

/// Lagrange interpolation with MultiPoly values
fn lagrange_interpolate_multipoly(
    var: &str,
    points: &[(BigRational, MultiPoly)],
    other_vars: &[String],
) -> Option<MultiPoly> {
    if points.is_empty() {
        return None;
    }

    // Build full variable list: [var, other_vars...]
    let mut all_vars = vec![var.to_string()];
    all_vars.extend(other_vars.iter().cloned());

    let mut result = MultiPoly::zero(all_vars.clone());

    for (i, (xi, yi)) in points.iter().enumerate() {
        // Build Lagrange basis L_i = prod_{j≠i} (var - x_j) / (x_i - x_j)
        let mut basis_numer = MultiPoly::one(all_vars.clone());
        let mut basis_denom = BigRational::one();

        for (j, (xj, _)) in points.iter().enumerate() {
            if i == j {
                continue;
            }

            // Numerator: multiply by (var - x_j)
            // Create (var - x_j) as MultiPoly
            let var_poly = var_minus_const(&all_vars, var, xj);
            basis_numer = basis_numer.mul(&var_poly, &PolyBudget::default()).ok()?;

            // Denominator: (x_i - x_j)
            let diff = xi - xj;
            if diff.is_zero() {
                return None;
            }
            basis_denom *= diff;
        }

        // Scale basis by 1/denom
        let inv_denom = BigRational::one() / basis_denom;
        basis_numer = basis_numer.mul_scalar(&inv_denom);

        // Embed yi into all_vars
        let yi_embedded = embed_multipoly(yi, &all_vars);

        // Multiply basis * yi_embedded and add to result
        let term = basis_numer.mul(&yi_embedded, &PolyBudget::default()).ok()?;
        result = result.add(&term).ok()?;
    }

    Some(result)
}

/// Create MultiPoly for (var - c)
fn var_minus_const(vars: &[String], var: &str, c: &BigRational) -> MultiPoly {
    let var_idx = vars.iter().position(|v| v == var).unwrap_or(0);
    let mut map = BTreeMap::new();

    // Constant term: -c
    if !c.is_zero() {
        map.insert(vec![0u32; vars.len()], -c.clone());
    }

    // Variable term: +var
    let mut mono = vec![0u32; vars.len()];
    mono[var_idx] = 1;
    map.insert(mono, BigRational::one());

    MultiPoly::from_map(vars.to_vec(), map)
}

/// Embed MultiPoly into larger variable space
fn embed_multipoly(p: &MultiPoly, target_vars: &[String]) -> MultiPoly {
    if p.vars.is_empty() {
        // Constant: embed into target_vars as constant term
        if let Some(c) = p.constant_value() {
            let mut map = BTreeMap::new();
            if !c.is_zero() {
                map.insert(vec![0u32; target_vars.len()], c);
            }
            return MultiPoly::from_map(target_vars.to_vec(), map);
        }
    }

    // Map old var indices to new indices
    let mapping: Vec<Option<usize>> = p
        .vars
        .iter()
        .map(|v| target_vars.iter().position(|tv| tv == v))
        .collect();

    let mut map = BTreeMap::new();
    for (coeff, mono) in &p.terms {
        let mut new_mono = vec![0u32; target_vars.len()];
        for (old_idx, &exp) in mono.iter().enumerate() {
            if let Some(Some(new_idx)) = mapping.get(old_idx) {
                new_mono[*new_idx] = exp;
            }
        }
        let entry = map.entry(new_mono).or_insert_with(BigRational::zero);
        *entry = entry.clone() + coeff.clone();
    }

    MultiPoly::from_map(target_vars.to_vec(), map)
}

/// Build GCD MultiPoly from coefficients
fn build_gcd_multivar(
    all_vars: &[String],
    main_var: VarIdx,
    _params: &[VarIdx],
    coeffs: &[MultiPoly],
) -> Option<MultiPoly> {
    let mut map = BTreeMap::new();

    for (k, coeff_poly) in coeffs.iter().enumerate() {
        // Embed coeff_poly (in param vars) into all_vars
        let coeff_embedded = embed_multipoly(coeff_poly, all_vars);

        for (c, mono) in &coeff_embedded.terms {
            if c.is_zero() {
                continue;
            }
            // Add main^k to the monomial
            let mut new_mono = mono.clone();
            if new_mono.len() <= main_var {
                new_mono.resize(all_vars.len(), 0);
            }
            new_mono[main_var] += k as u32;

            let entry = map.entry(new_mono).or_insert_with(BigRational::zero);
            *entry = entry.clone() + c.clone();
        }
    }

    Some(MultiPoly::from_map(all_vars.to_vec(), map))
}
