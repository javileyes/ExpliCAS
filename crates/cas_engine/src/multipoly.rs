//! Sparse Multivariate Polynomial over Q
//!
//! Representation: terms stored as (coefficient, monomial) pairs.
//! Monomial = Vec<u32> of exponents aligned with vars vector.
//! Uses BTreeMap internally for normalization, Vec for storage.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::collections::{BTreeMap, BTreeSet};

use cas_ast::{Context, Expr, ExprId};

// =============================================================================
// Types
// =============================================================================

pub type VarIdx = usize;
pub type Exp = u32;

/// Monomial = vector of exponents aligned with `vars`
/// Invariant: monomial.len() == vars.len()
pub type Monomial = Vec<Exp>;

/// Term: (coeff != 0, monomial)
pub type Term = (BigRational, Monomial);

/// Limits to avoid explosion during conversion/operations
#[derive(Clone, Debug)]
pub struct PolyBudget {
    pub max_terms: usize,
    pub max_total_degree: u32,
    /// Maximum exponent for Pow(sum, n) conversion. Prevents expensive expansions.
    pub max_pow_exp: u32,
}

impl Default for PolyBudget {
    fn default() -> Self {
        Self {
            max_terms: 200,
            max_total_degree: 32,
            max_pow_exp: 2, // Conservative: only expand x^0, x^1, x^2 during GCD
        }
    }
}

/// Error types for polynomial operations
#[derive(Clone, Debug)]
pub enum PolyError {
    NonPolynomial,
    BadExponent,
    NonConstantDivision,
    BudgetExceeded,
    VarMismatch,
}

impl std::fmt::Display for PolyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolyError::NonPolynomial => write!(f, "expression is not a polynomial over Q"),
            PolyError::BadExponent => write!(f, "non-integer or negative exponent"),
            PolyError::NonConstantDivision => write!(f, "division by non-constant expression"),
            PolyError::BudgetExceeded => write!(f, "budget exceeded"),
            PolyError::VarMismatch => write!(f, "variable mismatch"),
        }
    }
}

impl std::error::Error for PolyError {}

// =============================================================================
// MultiPoly struct
// =============================================================================

/// Sparse multivariate polynomial over Q
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiPoly {
    /// Ordered variable list (lexicographic)
    pub vars: Vec<String>,
    /// Terms in lex order by monomial, no zero coefficients
    pub terms: Vec<Term>,
}

impl MultiPoly {
    /// Create zero polynomial
    pub fn zero(vars: Vec<String>) -> Self {
        Self {
            vars,
            terms: Vec::new(),
        }
    }

    /// Create constant 1
    pub fn one(vars: Vec<String>) -> Self {
        let mono = vec![0; vars.len()];
        Self {
            vars,
            terms: vec![(BigRational::one(), mono)],
        }
    }

    /// Create from constant (no variables)
    pub fn from_const(c: BigRational) -> Self {
        if c.is_zero() {
            Self::zero(Vec::new())
        } else {
            Self {
                vars: Vec::new(),
                terms: vec![(c, Vec::new())],
            }
        }
    }

    /// Create monomial x_i (single variable with exp 1)
    pub fn from_var(vars: Vec<String>, var: &str) -> Result<Self, PolyError> {
        let idx = vars
            .iter()
            .position(|v| v == var)
            .ok_or(PolyError::VarMismatch)?;
        let mut mono = vec![0; vars.len()];
        mono[idx] = 1;
        Ok(Self {
            vars,
            terms: vec![(BigRational::one(), mono)],
        })
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Check if constant 1
    pub fn is_one(&self) -> bool {
        self.terms.len() == 1
            && self.terms[0].0 == BigRational::one()
            && self.terms[0].1.iter().all(|&e| e == 0)
    }

    /// Check if constant (possibly 0)
    pub fn is_constant(&self) -> bool {
        self.terms.is_empty() || (self.terms.len() == 1 && self.terms[0].1.iter().all(|&e| e == 0))
    }

    /// Get constant value if constant
    pub fn constant_value(&self) -> Option<BigRational> {
        if self.is_zero() {
            Some(BigRational::zero())
        } else if self.is_constant() {
            Some(self.terms[0].0.clone())
        } else {
            None
        }
    }

    /// Degree in a specific variable
    pub fn degree_in(&self, v: VarIdx) -> u32 {
        self.terms
            .iter()
            .map(|(_, m)| m.get(v).copied().unwrap_or(0))
            .max()
            .unwrap_or(0)
    }

    /// Total degree (max sum of exponents)
    pub fn total_degree(&self) -> u32 {
        self.terms
            .iter()
            .map(|(_, m)| m.iter().sum::<u32>())
            .max()
            .unwrap_or(0)
    }

    /// Leading term in lex order (first in sorted list)
    pub fn leading_term_lex(&self) -> Option<&Term> {
        self.terms.last() // terms are in ascending lex order, so last is largest
    }

    /// Number of terms
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }
}

// =============================================================================
// Normalization (combine like terms, remove zeros, sort lex)
// =============================================================================

impl MultiPoly {
    /// Build from BTreeMap (map has Monomial as key, coeff as value)
    pub fn from_map(vars: Vec<String>, map: BTreeMap<Monomial, BigRational>) -> Self {
        let terms: Vec<Term> = map
            .into_iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(m, c)| (c, m)) // Convert to (coeff, mono) for Term
            .collect();
        Self { vars, terms }
    }

    /// Convert to BTreeMap for operations (key=Monomial, value=coeff)
    pub fn to_map(&self) -> BTreeMap<Monomial, BigRational> {
        self.terms
            .iter()
            .map(|(c, m)| (m.clone(), c.clone()))
            .collect()
    }

    /// Normalize in place (if needed)
    pub fn normalize(&mut self) {
        let map = self.to_map();
        self.terms = map
            .into_iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(m, c)| (c, m))
            .collect();
    }
}

// =============================================================================
// Arithmetic
// =============================================================================

impl MultiPoly {
    /// Negate
    pub fn neg(&self) -> Self {
        let terms = self
            .terms
            .iter()
            .map(|(c, m)| (-c.clone(), m.clone()))
            .collect();
        Self {
            vars: self.vars.clone(),
            terms,
        }
    }

    /// Add two polynomials (must have same vars)
    pub fn add(&self, other: &Self) -> Result<Self, PolyError> {
        if self.vars != other.vars {
            return Err(PolyError::VarMismatch);
        }
        let mut map = self.to_map();
        for (c, m) in &other.terms {
            let entry = map.entry(m.clone()).or_insert_with(BigRational::zero);
            *entry = entry.clone() + c.clone();
        }
        Ok(Self::from_map(self.vars.clone(), map))
    }

    /// Subtract
    pub fn sub(&self, other: &Self) -> Result<Self, PolyError> {
        self.add(&other.neg())
    }

    /// Multiply by scalar
    pub fn mul_scalar(&self, k: &BigRational) -> Self {
        if k.is_zero() {
            return Self::zero(self.vars.clone());
        }
        let terms = self.terms.iter().map(|(c, m)| (c * k, m.clone())).collect();
        Self {
            vars: self.vars.clone(),
            terms,
        }
    }

    /// Multiply by monomial
    pub fn mul_monomial(&self, mono: &Monomial) -> Result<Self, PolyError> {
        if mono.len() != self.vars.len() {
            return Err(PolyError::VarMismatch);
        }
        let terms = self
            .terms
            .iter()
            .map(|(c, m)| {
                let new_mono: Monomial = m.iter().zip(mono.iter()).map(|(a, b)| a + b).collect();
                (c.clone(), new_mono)
            })
            .collect();
        Ok(Self {
            vars: self.vars.clone(),
            terms,
        })
    }

    /// Multiply two polynomials with budget
    pub fn mul(&self, other: &Self, budget: &PolyBudget) -> Result<Self, PolyError> {
        if self.vars != other.vars {
            return Err(PolyError::VarMismatch);
        }
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero(self.vars.clone()));
        }

        let mut map: BTreeMap<Monomial, BigRational> = BTreeMap::new();

        for (c1, m1) in &self.terms {
            for (c2, m2) in &other.terms {
                let new_coeff = c1 * c2;
                let new_mono: Monomial = m1.iter().zip(m2.iter()).map(|(a, b)| a + b).collect();

                // Check total degree
                let td: u32 = new_mono.iter().sum();
                if td > budget.max_total_degree {
                    return Err(PolyError::BudgetExceeded);
                }

                let entry = map.entry(new_mono).or_insert_with(BigRational::zero);
                *entry = entry.clone() + new_coeff;

                // Check term count
                if map.len() > budget.max_terms {
                    return Err(PolyError::BudgetExceeded);
                }
            }
        }

        Ok(Self::from_map(self.vars.clone(), map))
    }

    /// Fast multiplication using HashMap accumulation.
    /// Much faster than mul() for large polynomials (O(1) amortized per term vs O(log n)).
    /// Result is still normalized (sorted, no zero coeffs).
    pub fn mul_fast(&self, other: &Self, budget: &PolyBudget) -> Result<Self, PolyError> {
        use std::collections::HashMap;

        if self.vars != other.vars {
            return Err(PolyError::VarMismatch);
        }
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero(self.vars.clone()));
        }

        // Estimate result size and pre-allocate
        let estimated_terms = self.terms.len() * other.terms.len() / 2;
        let mut map: HashMap<Monomial, BigRational> = HashMap::with_capacity(estimated_terms);

        for (c1, m1) in &self.terms {
            for (c2, m2) in &other.terms {
                let new_coeff = c1 * c2;
                let new_mono: Monomial = m1.iter().zip(m2.iter()).map(|(a, b)| a + b).collect();

                // Check total degree
                let td: u32 = new_mono.iter().sum();
                if td > budget.max_total_degree {
                    return Err(PolyError::BudgetExceeded);
                }

                // O(1) amortized insertion
                map.entry(new_mono)
                    .and_modify(|c| *c = c.clone() + &new_coeff)
                    .or_insert(new_coeff);

                // Check term count (less frequently to reduce overhead)
                if map.len() > budget.max_terms {
                    return Err(PolyError::BudgetExceeded);
                }
            }
        }

        // Convert to sorted Vec, remove zeros, swap tuple order (HashMap has (mono, coeff), Term is (coeff, mono))
        let mut terms: Vec<Term> = map
            .into_iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(mono, coeff)| (coeff, mono))
            .collect();
        terms.sort_by(|a, b| a.1.cmp(&b.1)); // Sort by monomial (lex)

        Ok(Self {
            vars: self.vars.clone(),
            terms,
        })
    }
}

// =============================================================================
// Content / Primitive Part / Monomial GCD (Layer 1)
// =============================================================================

/// GCD of two BigRational (as fractions)
fn gcd_bigrational(a: &BigRational, b: &BigRational) -> BigRational {
    if a.is_zero() {
        return b.abs();
    }
    if b.is_zero() {
        return a.abs();
    }
    // gcd(a/b, c/d) = gcd(a*d, c*b) / (b*d)
    // But for content, we want gcd of numerators / lcm of denominators
    let num_gcd = a.numer().gcd(b.numer());
    let den_lcm = lcm_bigint(a.denom(), b.denom());
    BigRational::new(num_gcd, den_lcm)
}

fn lcm_bigint(a: &BigInt, b: &BigInt) -> BigInt {
    if a.is_zero() || b.is_zero() {
        return BigInt::zero();
    }
    let g = a.gcd(b);
    (a / &g) * b
}

impl MultiPoly {
    /// Content: GCD of all coefficients
    pub fn content(&self) -> BigRational {
        if self.is_zero() {
            return BigRational::zero();
        }
        let mut g = BigRational::zero();
        for (c, _) in &self.terms {
            g = gcd_bigrational(&g, c);
        }
        // Ensure positive
        if g < BigRational::zero() {
            g = -g;
        }
        g
    }

    /// Primitive part: self / content
    pub fn primitive_part(&self) -> (BigRational, Self) {
        let c = self.content();
        if c.is_zero() || c.is_one() {
            return (c, self.clone());
        }
        let pp = self.div_scalar_exact(&c).unwrap_or_else(|| self.clone());
        (c, pp)
    }

    /// Divide by scalar (exact)
    pub fn div_scalar_exact(&self, k: &BigRational) -> Option<Self> {
        if k.is_zero() {
            return None;
        }
        let terms = self.terms.iter().map(|(c, m)| (c / k, m.clone())).collect();
        Some(Self {
            vars: self.vars.clone(),
            terms,
        })
    }

    /// Monomial GCD of self: min exponent per variable across all terms
    pub fn monomial_gcd(&self) -> Monomial {
        if self.is_zero() || self.vars.is_empty() {
            return vec![0; self.vars.len()];
        }
        let n = self.vars.len();
        let mut result = vec![u32::MAX; n];
        for (_, m) in &self.terms {
            for i in 0..n {
                result[i] = result[i].min(m[i]);
            }
        }
        // Replace MAX with 0 (shouldn't happen if non-empty)
        result.iter_mut().for_each(|e| {
            if *e == u32::MAX {
                *e = 0;
            }
        });
        result
    }

    /// Monomial GCD between two polys
    pub fn monomial_gcd_with(&self, other: &Self) -> Result<Monomial, PolyError> {
        if self.vars != other.vars {
            return Err(PolyError::VarMismatch);
        }
        let mg1 = self.monomial_gcd();
        let mg2 = other.monomial_gcd();
        Ok(mg1
            .iter()
            .zip(mg2.iter())
            .map(|(a, b)| (*a).min(*b))
            .collect())
    }

    /// Divide by monomial (exact - all terms must be divisible)
    pub fn div_monomial_exact(&self, mono: &Monomial) -> Option<Self> {
        if mono.len() != self.vars.len() {
            return None;
        }
        // Check all terms are divisible
        for (_, m) in &self.terms {
            for (i, &e) in mono.iter().enumerate() {
                if m[i] < e {
                    return None;
                }
            }
        }
        let terms = self
            .terms
            .iter()
            .map(|(c, m)| {
                let new_mono: Monomial = m.iter().zip(mono.iter()).map(|(a, b)| a - b).collect();
                (c.clone(), new_mono)
            })
            .collect();
        Some(Self {
            vars: self.vars.clone(),
            terms,
        })
    }
}

// =============================================================================
// Exact Division (sparse multivariate)
// =============================================================================

impl MultiPoly {
    /// Exact division: self / divisor
    /// Returns None if doesn't divide exactly
    pub fn div_exact(&self, divisor: &Self) -> Option<Self> {
        if divisor.is_zero() {
            return None;
        }
        if self.is_zero() {
            return Some(Self::zero(self.vars.clone()));
        }
        if self.vars != divisor.vars {
            return None;
        }

        // Handle constant divisor
        if divisor.is_constant() {
            let c = divisor.constant_value()?;
            return self.div_scalar_exact(&c);
        }

        // Sparse division algorithm (lex order)
        let n_vars = self.vars.len();
        let mut remainder = self.to_map();
        let mut quotient: BTreeMap<Monomial, BigRational> = BTreeMap::new();

        // Leading term of divisor (largest in lex order)
        let (d_coeff, d_mono) = divisor.terms.last()?;

        while !remainder.is_empty() {
            // Get largest remaining term
            let (r_mono, r_coeff) = remainder.iter().next_back()?;
            let r_mono = r_mono.clone();
            let r_coeff = r_coeff.clone();

            // Check if divisor's leading monomial divides remainder's leading monomial
            let mut can_divide = true;
            let mut q_mono = vec![0u32; n_vars];
            for i in 0..n_vars {
                if r_mono[i] < d_mono[i] {
                    can_divide = false;
                    break;
                }
                q_mono[i] = r_mono[i] - d_mono[i];
            }

            if !can_divide {
                // Cannot divide - not exact
                return None;
            }

            // Quotient term
            let q_coeff = &r_coeff / d_coeff;

            // Subtract divisor * q_term from remainder
            for (dc, dm) in &divisor.terms {
                let sub_mono: Monomial = dm.iter().zip(q_mono.iter()).map(|(a, b)| a + b).collect();
                let sub_coeff = dc * &q_coeff;
                let entry = remainder.entry(sub_mono).or_insert_with(BigRational::zero);
                *entry = entry.clone() - sub_coeff;
            }

            // Clean zeros
            remainder.retain(|_, c| !c.is_zero());

            // Add to quotient
            let qe = quotient.entry(q_mono).or_insert_with(BigRational::zero);
            *qe = qe.clone() + q_coeff;
        }

        Some(Self::from_map(self.vars.clone(), quotient))
    }
}

// =============================================================================
// Layer 2: Evaluation, Interpolation, Heuristic GCD
// =============================================================================

use crate::polynomial::Polynomial;

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

impl MultiPoly {
    /// Evaluate a variable and adjust remaining indices for main/param vars
    fn eval_var_adjusted(
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

// =============================================================================
// AST Conversion
// =============================================================================

/// Collect all variable names from expression
pub fn collect_poly_vars(ctx: &Context, expr: ExprId) -> BTreeSet<String> {
    let mut vars = BTreeSet::new();
    collect_vars_recursive(ctx, expr, &mut vars);
    vars
}

fn collect_vars_recursive(ctx: &Context, expr: ExprId, vars: &mut BTreeSet<String>) {
    match ctx.get(expr) {
        Expr::Variable(name) => {
            vars.insert(name.clone());
        }
        Expr::Number(_) | Expr::Constant(_) => {}
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            collect_vars_recursive(ctx, *a, vars);
            collect_vars_recursive(ctx, *b, vars);
        }
        Expr::Neg(a) => {
            collect_vars_recursive(ctx, *a, vars);
        }
        _ => {} // Functions, matrices, etc. - ignore
    }
}

/// Convert expression to MultiPoly
pub fn multipoly_from_expr(
    ctx: &Context,
    expr: ExprId,
    budget: &PolyBudget,
) -> Result<MultiPoly, PolyError> {
    // Collect and sort variables
    let vars_set = collect_poly_vars(ctx, expr);
    let vars: Vec<String> = vars_set.into_iter().collect();

    // Convert
    from_expr_recursive(ctx, expr, &vars, budget)
}

fn from_expr_recursive(
    ctx: &Context,
    expr: ExprId,
    vars: &[String],
    budget: &PolyBudget,
) -> Result<MultiPoly, PolyError> {
    match ctx.get(expr) {
        Expr::Number(n) => Ok(MultiPoly {
            vars: vars.to_vec(),
            terms: if n.is_zero() {
                vec![]
            } else {
                vec![(n.clone(), vec![0; vars.len()])]
            },
        }),

        Expr::Variable(name) => {
            let idx = vars
                .iter()
                .position(|v| v == name)
                .ok_or(PolyError::NonPolynomial)?;
            let mut mono = vec![0; vars.len()];
            mono[idx] = 1;
            Ok(MultiPoly {
                vars: vars.to_vec(),
                terms: vec![(BigRational::one(), mono)],
            })
        }

        Expr::Neg(a) => {
            let p = from_expr_recursive(ctx, *a, vars, budget)?;
            Ok(p.neg())
        }

        Expr::Add(a, b) => {
            let pa = from_expr_recursive(ctx, *a, vars, budget)?;
            let pb = from_expr_recursive(ctx, *b, vars, budget)?;
            let result = pa.add(&pb)?;
            check_budget(&result, budget)?;
            Ok(result)
        }

        Expr::Sub(a, b) => {
            let pa = from_expr_recursive(ctx, *a, vars, budget)?;
            let pb = from_expr_recursive(ctx, *b, vars, budget)?;
            let result = pa.sub(&pb)?;
            check_budget(&result, budget)?;
            Ok(result)
        }

        Expr::Mul(a, b) => {
            let pa = from_expr_recursive(ctx, *a, vars, budget)?;
            let pb = from_expr_recursive(ctx, *b, vars, budget)?;
            pa.mul(&pb, budget)
        }

        Expr::Div(a, b) => {
            let pa = from_expr_recursive(ctx, *a, vars, budget)?;
            let pb = from_expr_recursive(ctx, *b, vars, budget)?;
            // Only allow division by constants
            if let Some(c) = pb.constant_value() {
                if c.is_zero() {
                    return Err(PolyError::NonPolynomial);
                }
                Ok(pa.mul_scalar(&(BigRational::one() / c)))
            } else {
                Err(PolyError::NonConstantDivision)
            }
        }

        Expr::Pow(base, exp) => {
            // Exponent must be non-negative integer constant
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() && *n >= BigRational::zero() {
                    let e: u32 = n
                        .to_integer()
                        .try_into()
                        .map_err(|_| PolyError::BadExponent)?;

                    // Check if exponent exceeds budget for Pow(sum, n) expansion
                    // Only apply budget check if base is a sum (Add) - constants/vars are cheap
                    if e > budget.max_pow_exp && matches!(ctx.get(*base), Expr::Add(_, _)) {
                        return Err(PolyError::BudgetExceeded);
                    }

                    let pb = from_expr_recursive(ctx, *base, vars, budget)?;
                    return pow_poly(&pb, e, budget);
                }
            }
            Err(PolyError::BadExponent)
        }

        Expr::Constant(_) => {
            // Constants like pi, e - treat as non-polynomial
            Err(PolyError::NonPolynomial)
        }

        _ => Err(PolyError::NonPolynomial),
    }
}

fn pow_poly(p: &MultiPoly, exp: u32, budget: &PolyBudget) -> Result<MultiPoly, PolyError> {
    if exp == 0 {
        return Ok(MultiPoly::one(p.vars.clone()));
    }
    if exp == 1 {
        return Ok(p.clone());
    }

    // Binary exponentiation
    let mut result = MultiPoly::one(p.vars.clone());
    let mut base = p.clone();
    let mut e = exp;

    while e > 0 {
        if e & 1 == 1 {
            result = result.mul(&base, budget)?;
        }
        e >>= 1;
        if e > 0 {
            base = base.mul(&base, budget)?;
        }
    }

    Ok(result)
}

fn check_budget(p: &MultiPoly, budget: &PolyBudget) -> Result<(), PolyError> {
    if p.num_terms() > budget.max_terms {
        return Err(PolyError::BudgetExceeded);
    }
    if p.total_degree() > budget.max_total_degree {
        return Err(PolyError::BudgetExceeded);
    }
    Ok(())
}

/// Convert MultiPoly back to expression
pub fn multipoly_to_expr(p: &MultiPoly, ctx: &mut Context) -> ExprId {
    if p.is_zero() {
        return ctx.num(0);
    }

    let mut term_exprs: Vec<ExprId> = Vec::new();

    for (coeff, mono) in &p.terms {
        let term = build_term_expr(ctx, coeff, mono, &p.vars);
        term_exprs.push(term);
    }

    // Sum all terms
    let mut result = term_exprs[0];
    for &t in &term_exprs[1..] {
        result = ctx.add_raw(Expr::Add(result, t));
    }

    result
}

fn build_term_expr(
    ctx: &mut Context,
    coeff: &BigRational,
    mono: &Monomial,
    vars: &[String],
) -> ExprId {
    // Build monomial part: x^a * y^b * ...
    let mut factors: Vec<ExprId> = Vec::new();

    for (i, &exp) in mono.iter().enumerate() {
        if exp > 0 {
            let var_expr = ctx.add(Expr::Variable(vars[i].clone()));
            if exp == 1 {
                factors.push(var_expr);
            } else {
                let exp_expr = ctx.num(exp as i64);
                factors.push(ctx.add_raw(Expr::Pow(var_expr, exp_expr)));
            }
        }
    }

    // Combine monomial factors
    let mono_expr = if factors.is_empty() {
        None
    } else {
        let mut m = factors[0];
        for &f in &factors[1..] {
            m = ctx.add_raw(Expr::Mul(m, f));
        }
        Some(m)
    };

    // Combine with coefficient
    if coeff.is_one() {
        mono_expr.unwrap_or_else(|| ctx.num(1))
    } else if *coeff == -BigRational::one() {
        let m = mono_expr.unwrap_or_else(|| ctx.num(1));
        ctx.add_raw(Expr::Neg(m))
    } else {
        let c_expr = ctx.add_raw(Expr::Number(coeff.clone()));
        if let Some(m) = mono_expr {
            ctx.add_raw(Expr::Mul(c_expr, m))
        } else {
            c_expr
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    fn parse_to_multipoly(input: &str) -> MultiPoly {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx).expect("parse failed");
        multipoly_from_expr(&ctx, expr, &PolyBudget::default()).expect("conversion failed")
    }

    #[test]
    fn test_from_expr_simple() {
        let p = parse_to_multipoly("x + 1");
        assert_eq!(p.vars, vec!["x".to_string()]);
        assert_eq!(p.num_terms(), 2);
    }

    #[test]
    fn test_from_expr_multivar() {
        let p = parse_to_multipoly("x*y + x + y + 1");
        assert_eq!(p.vars, vec!["x".to_string(), "y".to_string()]);
        assert_eq!(p.num_terms(), 4);
    }

    #[test]
    fn test_content() {
        let p = parse_to_multipoly("2*x + 4*y");
        let c = p.content();
        assert_eq!(c, BigRational::from_integer(2.into()));
    }

    #[test]
    fn test_monomial_gcd() {
        let p = parse_to_multipoly("x^2*y + x^3*y^2");
        let mg = p.monomial_gcd();
        assert_eq!(mg, vec![2, 1]); // x^2 * y
    }

    #[test]
    fn test_div_monomial_exact() {
        let p = parse_to_multipoly("x^2*y + x^3*y^2");
        let mg = p.monomial_gcd();
        let q = p.div_monomial_exact(&mg).expect("should divide");
        // Result: 1 + x*y
        assert_eq!(q.num_terms(), 2);
    }

    #[test]
    fn test_div_exact() {
        let p = parse_to_multipoly("x^2 - 1");
        let d = parse_to_multipoly("x - 1");
        let q = p.div_exact(&d).expect("should divide exactly");
        // Result: x + 1
        assert_eq!(q.num_terms(), 2);
    }

    #[test]
    fn test_div_exact_fails() {
        let p = parse_to_multipoly("x^2 + 1");
        let d = parse_to_multipoly("x - 1");
        assert!(p.div_exact(&d).is_none());
    }

    #[test]
    fn test_mul() {
        let p = parse_to_multipoly("x + 1");
        let q = parse_to_multipoly("x - 1");
        let r = p.mul(&q, &PolyBudget::default()).expect("mul");
        // Result: x^2 - 1
        assert_eq!(r.num_terms(), 2);
    }
}
