//! Expr → MultiPolyModP converter for poly_gcd_modp REPL functions.
//!
//! Converts symbolic expressions to mod-p polynomials for algebraic operations.

use crate::modp::{add_mod, inv_mod, mul_mod, neg_mod};
use crate::mono::MAX_VARS;
use crate::multipoly_modp::{build_linear_pow_direct, MultiPolyModP};
use cas_ast::{Context, Expr, ExprId};
use num_traits::ToPrimitive;
use rustc_hash::FxHashMap;

/// Budget for Expr → MultiPolyModP conversion
#[derive(Clone, Debug)]
pub struct PolyModpBudget {
    pub max_vars: usize,
    pub max_terms: usize,
    pub max_total_degree: u32,
    pub max_pow_exp: u32,
}

impl Default for PolyModpBudget {
    fn default() -> Self {
        Self {
            max_vars: MAX_VARS, // 8
            max_terms: 200_000,
            max_total_degree: 100,
            max_pow_exp: 100,
        }
    }
}

/// Variable name → index mapping
#[derive(Clone, Debug, Default)]
pub struct VarTable {
    pub names: Vec<String>,
    map: FxHashMap<String, usize>,
}

impl VarTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get index for a variable, inserting if new
    pub fn get_or_insert(&mut self, name: &str) -> Option<usize> {
        if let Some(&idx) = self.map.get(name) {
            return Some(idx);
        }
        if self.names.len() >= MAX_VARS {
            return None; // Too many variables
        }
        let idx = self.names.len();
        self.names.push(name.to_string());
        self.map.insert(name.to_string(), idx);
        Some(idx)
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// Error during conversion
#[derive(Clone, Debug)]
pub enum PolyConvError {
    TooManyVariables,
    TooManyTerms,
    ExponentTooLarge,
    NegativeExponent,
    UnsupportedExpr(String),
    BadPrime(String), // e.g., denominator divisible by p
}

impl std::fmt::Display for PolyConvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolyConvError::TooManyVariables => write!(f, "too many variables (max {})", MAX_VARS),
            PolyConvError::TooManyTerms => write!(f, "too many terms"),
            PolyConvError::ExponentTooLarge => write!(f, "exponent too large"),
            PolyConvError::NegativeExponent => write!(f, "negative exponent not supported"),
            PolyConvError::UnsupportedExpr(s) => write!(f, "unsupported expression: {}", s),
            PolyConvError::BadPrime(s) => write!(f, "bad prime: {}", s),
        }
    }
}

/// Strip __hold() wrappers from expression
pub fn strip_hold(ctx: &Context, mut expr: ExprId) -> ExprId {
    loop {
        match ctx.get(expr) {
            Expr::Function(name, args) if name == "__hold" && args.len() == 1 => {
                expr = args[0];
            }
            _ => return expr,
        }
    }
}

/// Convert Expr to MultiPolyModP
pub fn expr_to_poly_modp(
    ctx: &Context,
    expr: ExprId,
    p: u64,
    budget: &PolyModpBudget,
    vars: &mut VarTable,
) -> Result<MultiPolyModP, PolyConvError> {
    let expr = strip_hold(ctx, expr);
    expr_to_poly_modp_inner(ctx, expr, p, budget, vars)
}

fn expr_to_poly_modp_inner(
    ctx: &Context,
    expr: ExprId,
    p: u64,
    budget: &PolyModpBudget,
    vars: &mut VarTable,
) -> Result<MultiPolyModP, PolyConvError> {
    match ctx.get(expr) {
        Expr::Number(n) => {
            // Convert rational to mod p
            let num = n.numer();
            let den = n.denom();

            // num mod p
            let num_u64 = bigint_to_modp(num, p);

            // den mod p - need inverse
            let den_u64 = bigint_to_modp(den, p);
            if den_u64 == 0 {
                return Err(PolyConvError::BadPrime(format!(
                    "denominator {} divisible by prime {}",
                    den, p
                )));
            }

            let den_inv = inv_mod(den_u64, p)
                .ok_or_else(|| PolyConvError::BadPrime("cannot invert denominator".into()))?;

            let val = mul_mod(num_u64, den_inv, p);
            Ok(MultiPolyModP::constant(val, p, MAX_VARS))
        }

        Expr::Variable(name) => {
            let idx = vars
                .get_or_insert(name)
                .ok_or(PolyConvError::TooManyVariables)?;
            if idx >= budget.max_vars {
                return Err(PolyConvError::TooManyVariables);
            }
            Ok(MultiPolyModP::var(idx, p, MAX_VARS))
        }

        // For Add/Sub/Neg: flatten iteratively to avoid stack overflow
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_) => {
            // Iteratively flatten the entire Add/Sub/Neg tree
            let mut flat_terms: Vec<(ExprId, i8)> = Vec::new();
            flatten_add(ctx, expr, 1, &mut flat_terms);

            // Convert each term (these are now non-Add leaves)
            let mut result = MultiPolyModP::zero(p, MAX_VARS);
            for (term_expr, sign) in flat_terms {
                let term_poly = convert_non_add_term(ctx, term_expr, p, budget, vars)?;
                let signed_poly = if sign < 0 { term_poly.neg() } else { term_poly };
                result = result.add(&signed_poly);
                check_terms(&result, budget)?;
            }
            Ok(result)
        }

        Expr::Mul(l, r) => {
            let left = expr_to_poly_modp_inner(ctx, *l, p, budget, vars)?;
            let right = expr_to_poly_modp_inner(ctx, *r, p, budget, vars)?;
            let result = left.mul(&right);
            check_terms(&result, budget)?;
            Ok(result)
        }

        Expr::Pow(base, exp) => {
            // Extract integer exponent
            let exp_val = get_nonneg_int_exp(ctx, *exp)?;
            if exp_val > budget.max_pow_exp {
                return Err(PolyConvError::ExponentTooLarge);
            }

            // Try fast-path for linear base
            if let Some(coeffs) = try_extract_linear(ctx, *base, p, vars) {
                if coeffs.len() <= MAX_VARS + 1 {
                    let poly = build_linear_pow_direct(&coeffs, exp_val, p, MAX_VARS);
                    check_terms(&poly, budget)?;
                    return Ok(poly);
                }
            }

            // Fallback: convert base and use pow
            let base_poly = expr_to_poly_modp_inner(ctx, *base, p, budget, vars)?;
            let result = base_poly.pow(exp_val);
            check_terms(&result, budget)?;
            Ok(result)
        }

        Expr::Function(name, args) => {
            // Handle __hold by stripping
            if name == "__hold" && args.len() == 1 {
                return expr_to_poly_modp_inner(ctx, args[0], p, budget, vars);
            }
            Err(PolyConvError::UnsupportedExpr(format!("function {}", name)))
        }

        Expr::Div(_, _) => Err(PolyConvError::UnsupportedExpr("division".into())),

        Expr::Constant(c) => Err(PolyConvError::UnsupportedExpr(format!("constant {:?}", c))),

        Expr::Matrix { .. } => Err(PolyConvError::UnsupportedExpr("matrix".into())),

        Expr::SessionRef(_) => Err(PolyConvError::UnsupportedExpr("session reference".into())),
    }
}

/// Convert a non-Add/Sub/Neg expression term to polynomial.
/// This is used after flatten_add to process individual terms.
fn convert_non_add_term(
    ctx: &Context,
    expr: ExprId,
    p: u64,
    budget: &PolyModpBudget,
    vars: &mut VarTable,
) -> Result<MultiPolyModP, PolyConvError> {
    let expr = strip_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Number(n) => {
            let num = n.numer();
            let den = n.denom();
            let num_u64 = bigint_to_modp(num, p);
            let den_u64 = bigint_to_modp(den, p);
            if den_u64 == 0 {
                return Err(PolyConvError::BadPrime(format!(
                    "denominator {} divisible by prime {}",
                    den, p
                )));
            }
            let den_inv = inv_mod(den_u64, p)
                .ok_or_else(|| PolyConvError::BadPrime("cannot invert denominator".into()))?;
            let val = mul_mod(num_u64, den_inv, p);
            Ok(MultiPolyModP::constant(val, p, MAX_VARS))
        }

        Expr::Variable(name) => {
            let idx = vars
                .get_or_insert(name)
                .ok_or(PolyConvError::TooManyVariables)?;
            if idx >= budget.max_vars {
                return Err(PolyConvError::TooManyVariables);
            }
            Ok(MultiPolyModP::var(idx, p, MAX_VARS))
        }

        Expr::Mul(l, r) => {
            let left = convert_non_add_term(ctx, *l, p, budget, vars)?;
            let right = convert_non_add_term(ctx, *r, p, budget, vars)?;
            let result = left.mul(&right);
            check_terms(&result, budget)?;
            Ok(result)
        }

        Expr::Pow(base, exp) => {
            let exp_val = get_nonneg_int_exp(ctx, *exp)?;
            if exp_val > budget.max_pow_exp {
                return Err(PolyConvError::ExponentTooLarge);
            }

            // Try fast-path for linear base
            if let Some(coeffs) = try_extract_linear(ctx, *base, p, vars) {
                if coeffs.len() <= MAX_VARS + 1 {
                    let poly = build_linear_pow_direct(&coeffs, exp_val, p, MAX_VARS);
                    check_terms(&poly, budget)?;
                    return Ok(poly);
                }
            }

            // Fallback: convert base and use pow
            let base_poly = convert_non_add_term(ctx, *base, p, budget, vars)?;
            let result = base_poly.pow(exp_val);
            check_terms(&result, budget)?;
            Ok(result)
        }

        Expr::Function(name, args) => {
            if name == "__hold" && args.len() == 1 {
                return convert_non_add_term(ctx, args[0], p, budget, vars);
            }
            Err(PolyConvError::UnsupportedExpr(format!("function {}", name)))
        }

        // Shouldn't hit these after flatten_add, but handle gracefully
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_) => {
            // Fall back to full conversion for nested Add/Sub/Neg
            expr_to_poly_modp_inner(ctx, expr, p, budget, vars)
        }

        Expr::Div(_, _) => Err(PolyConvError::UnsupportedExpr("division".into())),
        Expr::Constant(c) => Err(PolyConvError::UnsupportedExpr(format!("constant {:?}", c))),
        Expr::Matrix { .. } => Err(PolyConvError::UnsupportedExpr("matrix".into())),
        Expr::SessionRef(_) => Err(PolyConvError::UnsupportedExpr("session reference".into())),
    }
}

/// Check term count against budget
fn check_terms(poly: &MultiPolyModP, budget: &PolyModpBudget) -> Result<(), PolyConvError> {
    if poly.num_terms() > budget.max_terms {
        Err(PolyConvError::TooManyTerms)
    } else {
        Ok(())
    }
}

/// Extract non-negative integer exponent
fn get_nonneg_int_exp(ctx: &Context, expr: ExprId) -> Result<u32, PolyConvError> {
    match ctx.get(expr) {
        Expr::Number(n) => {
            if !n.is_integer() {
                return Err(PolyConvError::UnsupportedExpr("fractional exponent".into()));
            }
            let int = n.to_integer();
            if int < num_bigint::BigInt::from(0) {
                return Err(PolyConvError::NegativeExponent);
            }
            int.to_u32().ok_or(PolyConvError::ExponentTooLarge)
        }
        _ => Err(PolyConvError::UnsupportedExpr(
            "non-numeric exponent".into(),
        )),
    }
}

/// Try to extract linear terms from Add tree: returns [c0, c1, ..., cn]
/// where the expression is c0 + c1*x1 + c2*x2 + ...
fn try_extract_linear(
    ctx: &Context,
    expr: ExprId,
    p: u64,
    vars: &mut VarTable,
) -> Option<Vec<u64>> {
    let expr = strip_hold(ctx, expr);

    // Collect additive terms
    let mut terms = Vec::new();
    flatten_add(ctx, expr, 1, &mut terms);

    // Parse each term as: coefficient * variable or just coefficient
    let mut coeffs = vec![0u64; MAX_VARS + 1]; // [const, x0, x1, ..., x7]

    for (term_id, sign) in terms {
        if let Some((coeff, var_idx)) = parse_linear_term(ctx, term_id, p, vars) {
            let signed_coeff = if sign < 0 { neg_mod(coeff, p) } else { coeff };
            let idx = var_idx.unwrap_or(0); // 0 = constant (index shift)
            let slot = if var_idx.is_some() { idx + 1 } else { 0 };
            if slot >= coeffs.len() {
                return None; // Too many variables
            }
            coeffs[slot] = add_mod(coeffs[slot], signed_coeff, p);
        } else {
            return None; // Not a linear term
        }
    }

    // Don't trim - build_linear_pow_direct expects exactly MAX_VARS + 1 coefficients
    Some(coeffs)
}

/// Flatten Add/Sub tree with sign tracking (iterative to avoid stack overflow)
fn flatten_add(ctx: &Context, expr: ExprId, initial_sign: i8, out: &mut Vec<(ExprId, i8)>) {
    // Use explicit work stack instead of recursion for large trees
    let mut work: Vec<(ExprId, i8)> = vec![(strip_hold(ctx, expr), initial_sign)];

    while let Some((curr_expr, sign)) = work.pop() {
        match ctx.get(curr_expr) {
            Expr::Add(l, r) => {
                // Push right first so left is processed first (LIFO)
                work.push((strip_hold(ctx, *r), sign));
                work.push((strip_hold(ctx, *l), sign));
            }
            Expr::Sub(l, r) => {
                work.push((strip_hold(ctx, *r), -sign));
                work.push((strip_hold(ctx, *l), sign));
            }
            Expr::Neg(inner) => {
                work.push((strip_hold(ctx, *inner), -sign));
            }
            _ => {
                out.push((curr_expr, sign));
            }
        }
    }
}

/// Parse a single linear term: coeff * var or just coeff or just var
/// Returns (coefficient, Some(var_idx)) or (coefficient, None) for constant
fn parse_linear_term(
    ctx: &Context,
    expr: ExprId,
    p: u64,
    vars: &mut VarTable,
) -> Option<(u64, Option<usize>)> {
    let expr = strip_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Number(n) => {
            if !n.is_integer() {
                return None; // Fractional coefficient in linear base
            }
            let val = bigint_to_modp(n.numer(), p);
            Some((val, None))
        }
        Expr::Variable(name) => {
            let idx = vars.get_or_insert(name)?;
            Some((1, Some(idx)))
        }
        Expr::Mul(l, r) => {
            // Try coeff * var
            if let (Some(coeff), Some(var_idx)) =
                (get_number(ctx, *l, p), get_var_idx(ctx, *r, vars))
            {
                return Some((coeff, Some(var_idx)));
            }
            // Try var * coeff
            if let (Some(var_idx), Some(coeff)) =
                (get_var_idx(ctx, *l, vars), get_number(ctx, *r, p))
            {
                return Some((coeff, Some(var_idx)));
            }
            None
        }
        Expr::Neg(inner) => {
            let (coeff, var_idx) = parse_linear_term(ctx, *inner, p, vars)?;
            Some((neg_mod(coeff, p), var_idx))
        }
        _ => None,
    }
}

fn get_number(ctx: &Context, expr: ExprId, p: u64) -> Option<u64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            return Some(bigint_to_modp(n.numer(), p));
        }
    }
    None
}

fn get_var_idx(ctx: &Context, expr: ExprId, vars: &mut VarTable) -> Option<usize> {
    if let Expr::Variable(name) = ctx.get(expr) {
        return vars.get_or_insert(name);
    }
    None
}

/// Convert BigInt to u64 mod p (handling negatives correctly)
fn bigint_to_modp(n: &num_bigint::BigInt, p: u64) -> u64 {
    use num_traits::Signed;
    let p_big = num_bigint::BigInt::from(p);
    let rem = n % &p_big;
    if rem.is_negative() {
        let rem_pos = (rem + &p_big) % &p_big;
        rem_pos.to_u64().unwrap_or(0)
    } else {
        rem.to_u64().unwrap_or(0)
    }
}

/// Default prime for mod-p operations (same as benchmark)
pub const DEFAULT_PRIME: u64 = 4503599627370449;

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_simple_var() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x", &mut ctx).unwrap();
        let mut vars = VarTable::new();
        let poly =
            expr_to_poly_modp(&ctx, expr, 17, &PolyModpBudget::default(), &mut vars).unwrap();
        assert_eq!(poly.num_terms(), 1);
        assert_eq!(vars.len(), 1);
    }

    #[test]
    fn test_linear_sum() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("1 + 2*x + 3*y", &mut ctx).unwrap();
        let mut vars = VarTable::new();
        let poly =
            expr_to_poly_modp(&ctx, expr, 17, &PolyModpBudget::default(), &mut vars).unwrap();
        assert_eq!(poly.num_terms(), 3);
    }

    #[test]
    fn test_pow_fast_path() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("(1 + x)^3", &mut ctx).unwrap();
        let mut vars = VarTable::new();
        let poly =
            expr_to_poly_modp(&ctx, expr, 17, &PolyModpBudget::default(), &mut vars).unwrap();
        // (1+x)^3 = 1 + 3x + 3x^2 + x^3
        assert_eq!(poly.num_terms(), 4);
    }
}
