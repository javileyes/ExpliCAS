//! Expr → MultiPolyModP converter for poly_gcd_modp REPL functions.
//!
//! Converts symbolic expressions to mod-p polynomials for algebraic operations.

use crate::modp::{add_mod, inv_mod, mul_mod, neg_mod};
use crate::mono::MAX_VARS;
use crate::multipoly_modp::{build_linear_pow_direct, MultiPolyModP};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::ToPrimitive;
use rustc_hash::FxHashMap;

/// Budget for Expr → MultiPolyModP conversion
#[derive(Clone, Debug)]
pub struct PolyModpBudget {
    pub max_vars: usize,
    pub max_terms: usize,
    #[allow(dead_code)] // Budget cap not yet enforced; kept for future degree-limit checks
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

    /// Get variable names in index order
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Get the index for a variable name, if it exists
    pub fn get_index(&self, name: &str) -> Option<usize> {
        self.map.get(name).copied()
    }

    /// Create a VarTable from a fixed list of names.
    pub fn from_names(names: &[String]) -> Self {
        let mut table = Self::new();
        for name in names {
            table.get_or_insert(name);
        }
        table
    }

    /// Unify two VarTables into a canonical unified table.
    ///
    /// Returns:
    /// - `unified`: The merged VarTable with lexicographically sorted variables
    /// - `remap_a`: Map from this table's indices to unified indices
    /// - `remap_b`: Map from other table's indices to unified indices
    ///
    /// Returns None if the unified table would exceed MAX_VARS.
    pub fn unify(&self, other: &VarTable) -> Option<(VarTable, Vec<usize>, Vec<usize>)> {
        // Collect all unique variable names
        let mut all_names: Vec<&str> = Vec::new();
        for name in &self.names {
            if !all_names.contains(&name.as_str()) {
                all_names.push(name);
            }
        }
        for name in &other.names {
            if !all_names.contains(&name.as_str()) {
                all_names.push(name);
            }
        }

        // Check if too many variables
        if all_names.len() > MAX_VARS {
            return None;
        }

        // Sort lexicographically for canonical ordering
        all_names.sort();

        // Build unified VarTable
        let mut unified = VarTable::new();
        for name in &all_names {
            unified.get_or_insert(name)?;
        }

        // Build remap vectors
        let remap_a: Vec<usize> = self
            .names
            .iter()
            .map(|n| unified.get_index(n))
            .collect::<Option<Vec<_>>>()?;

        let remap_b: Vec<usize> = other
            .names
            .iter()
            .map(|n| unified.get_index(n))
            .collect::<Option<Vec<_>>>()?;

        Some((unified, remap_a, remap_b))
    }

    /// Check if this VarTable has the same variables as another (same set, any order)
    pub fn same_variables(&self, other: &VarTable) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for name in &self.names {
            if !other.map.contains_key(name) {
                return false;
            }
        }
        true
    }

    /// Check if this VarTable has the same variables in the same order
    pub fn same_order(&self, other: &VarTable) -> bool {
        self.names == other.names
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

/// External resolver for `poly_result(id)` values.
pub trait PolyResultResolver {
    fn resolve_poly_result(
        &self,
        ctx: &Context,
        p: u64,
        vars: &mut VarTable,
        id_expr: ExprId,
    ) -> Result<MultiPolyModP, PolyConvError>;
}

/// Default resolver used by the pure converter (no engine-specific poly store).
pub struct NoPolyResultResolver;

impl PolyResultResolver for NoPolyResultResolver {
    fn resolve_poly_result(
        &self,
        _ctx: &Context,
        _p: u64,
        _vars: &mut VarTable,
        _id_expr: ExprId,
    ) -> Result<MultiPolyModP, PolyConvError> {
        Err(PolyConvError::UnsupportedExpr("poly_result".into()))
    }
}

/// Resolver backed by the thread-local `PolyStore`.
pub struct ThreadLocalPolyStoreResolver;

impl PolyResultResolver for ThreadLocalPolyStoreResolver {
    fn resolve_poly_result(
        &self,
        ctx: &Context,
        p: u64,
        vars: &mut VarTable,
        id_expr: ExprId,
    ) -> Result<MultiPolyModP, PolyConvError> {
        use crate::poly_store::{thread_local_get_for_materialize, PolyId};

        let id_u32: u32 = match ctx.get(id_expr) {
            Expr::Number(n) => n.to_integer().to_u32().ok_or_else(|| {
                PolyConvError::UnsupportedExpr("poly_result id not valid integer".into())
            })?,
            _ => {
                return Err(PolyConvError::UnsupportedExpr(
                    "poly_result arg must be integer".into(),
                ))
            }
        };

        let poly_id: PolyId = id_u32;
        let (meta, poly) = thread_local_get_for_materialize(poly_id).ok_or_else(|| {
            PolyConvError::UnsupportedExpr(format!("invalid poly_result({})", poly_id))
        })?;

        if poly.p != p {
            return Err(PolyConvError::BadPrime(format!(
                "poly_result modulus {} differs from requested {}",
                poly.p, p
            )));
        }

        for name in &meta.var_names {
            vars.get_or_insert(name)
                .ok_or(PolyConvError::TooManyVariables)?;
        }

        let remap: Vec<usize> = meta
            .var_names
            .iter()
            .map(|name| {
                vars.get_index(name).ok_or_else(|| {
                    PolyConvError::UnsupportedExpr(format!(
                        "VarTable missing variable '{}' after insert",
                        name
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(poly.remap(&remap, vars.len()))
    }
}

/// Strip __hold() wrappers from expression (multi-level)
/// Uses canonical implementation from cas_ast::hold
pub fn strip_hold(ctx: &Context, mut expr: ExprId) -> ExprId {
    loop {
        let unwrapped = cas_ast::hold::unwrap_hold(ctx, expr);
        if unwrapped == expr {
            return expr;
        }
        expr = unwrapped;
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
    expr_to_poly_modp_with_resolver(ctx, expr, p, budget, vars, &NoPolyResultResolver)
}

/// Convert Expr to MultiPolyModP with external `poly_result` resolution.
pub fn expr_to_poly_modp_with_resolver<R: PolyResultResolver>(
    ctx: &Context,
    expr: ExprId,
    p: u64,
    budget: &PolyModpBudget,
    vars: &mut VarTable,
    resolver: &R,
) -> Result<MultiPolyModP, PolyConvError> {
    let expr = strip_hold(ctx, expr);
    expr_to_poly_modp_inner(ctx, expr, p, budget, vars, resolver)
}

/// Convert Expr to MultiPolyModP, resolving `poly_result(id)` via thread-local store.
pub fn expr_to_poly_modp_with_store(
    ctx: &Context,
    expr: ExprId,
    p: u64,
    budget: &PolyModpBudget,
    vars: &mut VarTable,
) -> Result<MultiPolyModP, PolyConvError> {
    expr_to_poly_modp_with_resolver(ctx, expr, p, budget, vars, &ThreadLocalPolyStoreResolver)
}

fn expr_to_poly_modp_inner<R: PolyResultResolver>(
    ctx: &Context,
    expr: ExprId,
    p: u64,
    budget: &PolyModpBudget,
    vars: &mut VarTable,
    resolver: &R,
) -> Result<MultiPolyModP, PolyConvError> {
    // Early return for __hold - strip and recurse (using canonical helper)
    if let Some(inner) = cas_ast::hold::unwrap_hold_if_wrapped(ctx, expr) {
        return expr_to_poly_modp_inner(ctx, inner, p, budget, vars, resolver);
    }

    // Early return for poly_result via resolver.
    if let Expr::Function(name, args) = ctx.get(expr) {
        if ctx.is_builtin(*name, BuiltinFn::PolyResult) && args.len() == 1 {
            return resolver.resolve_poly_result(ctx, p, vars, args[0]);
        }
    }

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

        Expr::Variable(sym_id) => {
            let name = ctx.sym_name(*sym_id);
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
            collect_add_terms(ctx, expr, 1, &mut flat_terms);

            // Convert each term (these are now non-Add leaves)
            let mut result = MultiPolyModP::zero(p, MAX_VARS);
            for (term_expr, sign) in flat_terms {
                let term_poly = convert_non_add_term(ctx, term_expr, p, budget, vars, resolver)?;
                let signed_poly = if sign < 0 { term_poly.neg() } else { term_poly };
                result = result.add(&signed_poly);
                check_terms(&result, budget)?;
            }
            Ok(result)
        }

        Expr::Mul(l, r) => {
            let left = expr_to_poly_modp_inner(ctx, *l, p, budget, vars, resolver)?;
            let right = expr_to_poly_modp_inner(ctx, *r, p, budget, vars, resolver)?;
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
            let base_poly = expr_to_poly_modp_inner(ctx, *base, p, budget, vars, resolver)?;
            let result = base_poly.pow(exp_val);
            check_terms(&result, budget)?;
            Ok(result)
        }

        // __hold and poly_result are handled by early returns above
        Expr::Function(ref name, _) => {
            let fn_name = ctx.sym_name(*name);
            Err(PolyConvError::UnsupportedExpr(format!(
                "function {}",
                fn_name
            )))
        }

        Expr::Div(_, _) => Err(PolyConvError::UnsupportedExpr("division".into())),

        Expr::Constant(c) => Err(PolyConvError::UnsupportedExpr(format!("constant {:?}", c))),

        Expr::Matrix { .. } => Err(PolyConvError::UnsupportedExpr("matrix".into())),

        Expr::SessionRef(_) => Err(PolyConvError::UnsupportedExpr("session reference".into())),

        // Hold is stripped at function entry, but handle explicitly for exhaustiveness
        Expr::Hold(inner) => expr_to_poly_modp_inner(ctx, *inner, p, budget, vars, resolver),
    }
}

/// Convert a non-Add/Sub/Neg expression term to polynomial.
/// This is used after flatten_add to process individual terms.
fn convert_non_add_term<R: PolyResultResolver>(
    ctx: &Context,
    expr: ExprId,
    p: u64,
    budget: &PolyModpBudget,
    vars: &mut VarTable,
    resolver: &R,
) -> Result<MultiPolyModP, PolyConvError> {
    let expr = strip_hold(ctx, expr);

    // Early return for poly_result via resolver.
    if let Expr::Function(name, args) = ctx.get(expr) {
        if ctx.is_builtin(*name, BuiltinFn::PolyResult) && args.len() == 1 {
            return resolver.resolve_poly_result(ctx, p, vars, args[0]);
        }
    }

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

        Expr::Variable(sym_id) => {
            let name = ctx.sym_name(*sym_id);
            let idx = vars
                .get_or_insert(name)
                .ok_or(PolyConvError::TooManyVariables)?;
            if idx >= budget.max_vars {
                return Err(PolyConvError::TooManyVariables);
            }
            Ok(MultiPolyModP::var(idx, p, MAX_VARS))
        }

        Expr::Mul(l, r) => {
            let left = convert_non_add_term(ctx, *l, p, budget, vars, resolver)?;
            let right = convert_non_add_term(ctx, *r, p, budget, vars, resolver)?;
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
            let base_poly = convert_non_add_term(ctx, *base, p, budget, vars, resolver)?;
            let result = base_poly.pow(exp_val);
            check_terms(&result, budget)?;
            Ok(result)
        }

        // __hold is already stripped at function entry
        // poly_result is handled by early return above
        Expr::Function(ref name, _) => {
            let fn_name = ctx.sym_name(*name);
            Err(PolyConvError::UnsupportedExpr(format!(
                "function {}",
                fn_name
            )))
        }

        // Shouldn't hit these after flatten_add, but handle gracefully
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_) => {
            // Fall back to full conversion for nested Add/Sub/Neg
            expr_to_poly_modp_inner(ctx, expr, p, budget, vars, resolver)
        }

        Expr::Div(_, _) => Err(PolyConvError::UnsupportedExpr("division".into())),
        Expr::Constant(c) => Err(PolyConvError::UnsupportedExpr(format!("constant {:?}", c))),
        Expr::Matrix { .. } => Err(PolyConvError::UnsupportedExpr("matrix".into())),
        Expr::SessionRef(_) => Err(PolyConvError::UnsupportedExpr("session reference".into())),

        // Hold is stripped at function entry, but handle explicitly for exhaustiveness
        Expr::Hold(inner) => convert_non_add_term(ctx, *inner, p, budget, vars, resolver),
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
    collect_add_terms(ctx, expr, 1, &mut terms);

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

fn collect_add_terms(ctx: &Context, expr: ExprId, initial_sign: i8, out: &mut Vec<(ExprId, i8)>) {
    let mut stack = vec![(expr, initial_sign)];
    while let Some((node, sign)) = stack.pop() {
        let node = strip_hold(ctx, node);
        match ctx.get(node) {
            Expr::Add(l, r) => {
                stack.push((*r, sign));
                stack.push((*l, sign));
            }
            Expr::Sub(l, r) => {
                stack.push((*r, -sign));
                stack.push((*l, sign));
            }
            Expr::Neg(inner) => {
                stack.push((*inner, -sign));
            }
            _ => out.push((node, sign)),
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
        Expr::Variable(sym_id) => {
            let name = ctx.sym_name(*sym_id);
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
    if let Expr::Variable(sym_id) = ctx.get(expr) {
        let name = ctx.sym_name(*sym_id);
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

/// Convert a mod-p coefficient to symmetric signed representation.
///
/// Coefficients in [0, p/2] stay positive, and (p/2, p-1] become negative.
#[inline]
fn modp_to_signed(c: u64, p: u64) -> i128 {
    let half = p / 2;
    if c <= half {
        c as i128
    } else {
        (c as i128) - (p as i128)
    }
}

/// Convert MultiPolyModP back to an Expr (balanced Add tree).
pub fn multipoly_modp_to_expr(ctx: &mut Context, poly: &MultiPolyModP, vars: &VarTable) -> ExprId {
    use num_bigint::BigInt;
    use num_rational::BigRational;

    if poly.is_zero() {
        return ctx.num(0);
    }

    let p = poly.p;
    let mut term_exprs: Vec<ExprId> = Vec::with_capacity(poly.terms.len());

    for (mono, coeff) in &poly.terms {
        if *coeff == 0 {
            continue;
        }

        let signed_coeff = modp_to_signed(*coeff, p);
        let is_negative = signed_coeff < 0;
        let abs_coeff = signed_coeff.unsigned_abs();

        let mut factors: Vec<ExprId> = Vec::new();

        // Emit explicit coefficient unless it is implicit 1 for non-constant monomials.
        if abs_coeff != 1 || mono.is_constant() {
            factors.push(ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(
                abs_coeff,
            )))));
        }

        for (i, &exp) in mono.0.iter().enumerate() {
            if exp > 0 && i < vars.len() {
                let var_expr = ctx.var(&vars.names[i]);
                if exp == 1 {
                    factors.push(var_expr);
                } else {
                    let exp_expr =
                        ctx.add(Expr::Number(BigRational::from_integer((exp as i64).into())));
                    factors.push(ctx.add(Expr::Pow(var_expr, exp_expr)));
                }
            }
        }

        let term = if factors.is_empty() {
            ctx.num(1)
        } else if factors.len() == 1 {
            factors[0]
        } else {
            factors
                .into_iter()
                .reduce(|acc, f| ctx.add(Expr::Mul(acc, f)))
                .unwrap_or_else(|| ctx.num(1))
        };

        let final_term = if is_negative {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };

        term_exprs.push(final_term);
    }

    if term_exprs.is_empty() {
        ctx.num(0)
    } else {
        build_balanced_sum(ctx, &term_exprs)
    }
}

fn build_balanced_sum(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    match terms.len() {
        0 => ctx.num(0),
        1 => terms[0],
        2 => ctx.add(Expr::Add(terms[0], terms[1])),
        _ => {
            let mid = terms.len() / 2;
            let left = build_balanced_sum(ctx, &terms[..mid]);
            let right = build_balanced_sum(ctx, &terms[mid..]);
            ctx.add(Expr::Add(left, right))
        }
    }
}

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
