//! Fast multinomial expansion for linear polynomials raised to a power.
//!
//! Implements `(c0 + c1*x1 + c2*x2 + ... + cn*xn)^exp` directly using multinomial
//! coefficients. This is O(output_terms) instead of O(terms²) for repeated multiplication.
//!
//! Used by `expand()` as a fast-path for the common case of expanding sums of
//! linear terms.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

use cas_ast::{Context, Expr, ExprId};

/// Budget limits for multinomial expansion to prevent runaway computation.
#[derive(Clone, Copy, Debug)]
pub struct MultinomialExpandBudget {
    /// Maximum exponent allowed (e.g., 12)
    pub max_exp: u32,
    /// Maximum number of terms in the base (e.g., 16)
    pub max_base_terms: usize,
    /// Maximum number of distinct variables (e.g., 8)
    pub max_vars: usize,
    /// Maximum number of output terms (e.g., 100_000)
    pub max_output_terms: usize,
}

impl Default for MultinomialExpandBudget {
    fn default() -> Self {
        Self {
            max_exp: 100, // High limit - real constraint is max_output_terms
            max_base_terms: 16,
            max_vars: 8,
            max_output_terms: 100_000,
        }
    }
}

/// A linear term: coefficient * variable (or just coefficient if constant)
#[derive(Clone, Debug)]
struct LinearTerm {
    /// Coefficient (can be negative)
    coeff: BigRational,
    /// Variable ExprId, or None for constant term
    var: Option<ExprId>,
}

#[derive(Clone, Copy)]
enum TermSign {
    Pos,
    Neg,
}

impl TermSign {
    #[inline]
    fn negate(self) -> Self {
        match self {
            Self::Pos => Self::Neg,
            Self::Neg => Self::Pos,
        }
    }

    #[inline]
    fn is_negative(self) -> bool {
        matches!(self, Self::Neg)
    }
}

/// Monomial key: exponents for each variable in canonical order
#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct MonoKey(Vec<u16>);

/// Try to expand `base^exp` using fast multinomial direct method.
///
/// Returns Some(expanded_expr) if:
/// - base is a sum of linear terms (const + c_i * x_i)
/// - exp is a small positive integer
/// - output term count is within budget
///
/// Returns None if pattern doesn't match or budget exceeded (fall back to slow path).
pub fn try_expand_multinomial_direct(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    budget: &MultinomialExpandBudget,
) -> Option<ExprId> {
    // 1. Check exponent is small positive integer
    let n = match ctx.get(exp) {
        Expr::Number(num) => {
            if !num.is_integer() || num.is_negative() {
                return None;
            }
            num.to_integer().to_u32()?
        }
        _ => return None,
    };

    if n < 2 || n > budget.max_exp {
        return None;
    }

    // 2. Extract linear terms from base
    let terms = extract_linear_terms(ctx, base)?;
    let k = terms.len();

    if k < 2 || k > budget.max_base_terms {
        return None;
    }

    // Count distinct variables
    let var_count = terms.iter().filter(|t| t.var.is_some()).count();
    if var_count > budget.max_vars {
        return None;
    }

    // 3. Estimate output term count: C(n+k-1, k-1)
    let estimated = multinomial_term_count(n, k, budget.max_output_terms)?;
    if estimated > budget.max_output_terms {
        return None;
    }

    // 4. Build variable ordering (canonical, deterministic)
    let mut var_ids: Vec<ExprId> = terms.iter().filter_map(|t| t.var).collect();
    // Sort by expression ID index for determinism (stable u32 ordering)
    var_ids.sort_by_key(|id| id.index());
    var_ids.dedup();

    // Map variable ExprId -> index in var_ids
    let var_to_idx: FxHashMap<ExprId, usize> =
        var_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // 5. Precompute factorial table
    let fact = factorial_table(n);

    // 6. Precompute power tables for each coefficient
    let pow_tables: Vec<Vec<BigRational>> = terms
        .iter()
        .map(|t| pow_table_rational(&t.coeff, n))
        .collect();

    // 7. Enumerate compositions and accumulate terms
    let num_vars = var_ids.len();
    let mut acc: FxHashMap<MonoKey, BigRational> = FxHashMap::default();
    let mut composition = vec![0u32; k];

    enumerate_compositions(n, k, &mut composition, &mut |comp| {
        // Multinomial coefficient: n! / (k0! * k1! * ... * k_{k-1}!)
        let mcoeff = multinomial_coeff(&fact, n, comp);

        // Product of coefficient powers: Π (c_i)^(e_i)
        let mut coeff = BigRational::from(mcoeff);
        for (i, &e) in comp.iter().enumerate() {
            coeff *= &pow_tables[i][e as usize];
        }

        if coeff.is_zero() {
            return;
        }

        // Build monomial key: exponents for each variable
        let mut mono_exps = vec![0u16; num_vars];
        for (i, &e) in comp.iter().enumerate() {
            if let Some(var_id) = terms[i].var {
                if let Some(&idx) = var_to_idx.get(&var_id) {
                    mono_exps[idx] += e as u16;
                }
            }
        }
        let key = MonoKey(mono_exps);

        // Accumulate
        acc.entry(key).and_modify(|c| *c += &coeff).or_insert(coeff);
    });

    // 8. Convert accumulated terms to sorted vec (filter zeros)
    let mut terms_vec: Vec<(MonoKey, BigRational)> =
        acc.into_iter().filter(|(_, c)| !c.is_zero()).collect();

    // Sort by monomial (descending, typical polynomial order)
    terms_vec.sort_by(|a, b| b.0.cmp(&a.0));

    // 9. Emit polynomial as Expr
    let expanded = emit_polynomial_from_terms(ctx, &var_ids, terms_vec);

    // 10. Wrap in __hold() to prevent slow post-simplification traversal
    // The __hold barrier is unwrapped at eval boundary (engine.rs::unwrap_hold_top)
    let held = cas_ast::hold::wrap_hold(ctx, expanded);
    Some(held)
}

/// Collect additive terms into a vector of (summand, sign) pairs.
///
/// This helper is shape-independent and transparent to top-level `__hold(...)`.
fn collect_add_terms_signed(
    ctx: &Context,
    root: ExprId,
    initial_sign: TermSign,
    out: &mut Vec<(ExprId, TermSign)>,
) {
    let mut stack = Vec::with_capacity(16);
    stack.push((root, initial_sign));

    while let Some((id, sign)) = stack.pop() {
        let id = cas_ast::hold::unwrap_hold(ctx, id);
        match ctx.get(id) {
            Expr::Add(l, r) => {
                stack.push((*r, sign));
                stack.push((*l, sign));
            }
            Expr::Sub(l, r) => {
                stack.push((*r, sign.negate()));
                stack.push((*l, sign));
            }
            Expr::Neg(inner) => stack.push((*inner, sign.negate())),
            _ => out.push((id, sign)),
        }
    }
}

/// Extract linear terms from a sum expression.
///
/// Returns Some(vec) if all terms are linear: const, x, c*x, -c*x
/// Returns None if any term is non-linear (x*y, x^2, sin(x), etc.)
fn extract_linear_terms(ctx: &Context, base: ExprId) -> Option<Vec<LinearTerm>> {
    let mut summands = Vec::with_capacity(8);
    collect_add_terms_signed(ctx, base, TermSign::Pos, &mut summands);

    let mut terms = Vec::with_capacity(summands.len());
    for (s, sign) in summands {
        let mut term = parse_linear_atom(ctx, s)?;
        if sign.is_negative() {
            term.coeff = -term.coeff;
        }
        terms.push(term);
    }

    Some(terms)
}

/// Parse a single term as a coefficient × optional atom.
///
/// "Atom" is any non-numeric subexpression: variables, constants (π, e),
/// function calls (sin(x)), powers (x²), etc.  The multinomial expansion
/// is purely algebraic and does not care what the atom represents.
///
/// Accepts:
/// - Number:        (n, None)
/// - Atom:          (1, Some(atom))
/// - Neg(Atom):     (-1, Some(atom))
/// - Mul(Num, Atom) or Mul(Atom, Num): (n, Some(atom))
/// - Neg(Mul(Num, Atom)):              (-n, Some(atom))
fn parse_linear_atom(ctx: &Context, term: ExprId) -> Option<LinearTerm> {
    match ctx.get(term) {
        Expr::Number(n) => Some(LinearTerm {
            coeff: n.clone(),
            var: None,
        }),
        Expr::Neg(inner) => {
            let mut lt = parse_linear_atom(ctx, *inner)?;
            lt.coeff = -lt.coeff;
            Some(lt)
        }
        Expr::Mul(l, r) => {
            // Check for Number * Atom or Atom * Number
            match (ctx.get(*l), ctx.get(*r)) {
                (Expr::Number(n), _) if !matches!(ctx.get(*r), Expr::Number(_)) => {
                    Some(LinearTerm {
                        coeff: n.clone(),
                        var: Some(*r),
                    })
                }
                (_, Expr::Number(n)) if !matches!(ctx.get(*l), Expr::Number(_)) => {
                    Some(LinearTerm {
                        coeff: n.clone(),
                        var: Some(*l),
                    })
                }
                // Neg(Number) * Atom
                (Expr::Neg(inner), _) if !matches!(ctx.get(*r), Expr::Number(_)) => {
                    if let Expr::Number(n) = ctx.get(*inner) {
                        Some(LinearTerm {
                            coeff: -n.clone(),
                            var: Some(*r),
                        })
                    } else {
                        None // Mul with non-linear structure
                    }
                }
                _ => None, // Mul of two non-numeric things (e.g., x*y) — not linear
            }
        }
        // Pow with integer exponent (e.g., x^2) is a valid atom.
        // Pow with fractional exponent (e.g., 2^(1/2) = √2) is NOT: expanding
        // it produces nested Pow(Pow(base,frac),n) that doesn't fold, and it
        // interferes with the rationalization pipeline.
        Expr::Pow(_base, exp) => {
            let is_integer_exp = matches!(ctx.get(*exp), Expr::Number(n) if n.is_integer());
            if is_integer_exp {
                Some(LinearTerm {
                    coeff: BigRational::one(),
                    var: Some(term),
                })
            } else {
                None // fractional power → not a valid atom for multinomial expansion
            }
        }
        // Any other non-numeric expression: treat as opaque atom with coeff=1
        _ => Some(LinearTerm {
            coeff: BigRational::one(),
            var: Some(term),
        }),
    }
}

/// Estimate multinomial term count: C(n+k-1, k-1)
/// Returns None if overflow or exceeds max.
pub fn multinomial_term_count(n: u32, k: usize, max: usize) -> Option<usize> {
    // C(n+k-1, k-1) = (n+k-1)! / ((k-1)! * n!)
    // Use iterative calculation to avoid overflow
    let top = (n as usize) + k - 1;
    let bot = k - 1;

    let mut result: u128 = 1;
    for i in 0..bot {
        result = result * (top - i) as u128 / (i + 1) as u128;
        if result > max as u128 {
            return None;
        }
    }

    Some(result as usize)
}

/// Build factorial table up to n
fn factorial_table(n: u32) -> Vec<BigInt> {
    let mut fact = Vec::with_capacity((n + 1) as usize);
    fact.push(BigInt::one());
    for i in 1..=n {
        // INVARIANT: fact is never empty (initialized above)
        let prev = fact.last().unwrap_or(&BigInt::one()).clone();
        fact.push(prev * BigInt::from(i));
    }
    fact
}

/// Compute multinomial coefficient: n! / (k0! * k1! * ... * km!)
fn multinomial_coeff(fact: &[BigInt], n: u32, ks: &[u32]) -> BigInt {
    let mut denom = BigInt::one();
    for &ki in ks {
        denom *= &fact[ki as usize];
    }
    &fact[n as usize] / denom
}

/// Precompute power table: c^0, c^1, ..., c^n
fn pow_table_rational(c: &BigRational, n: u32) -> Vec<BigRational> {
    let mut table = Vec::with_capacity((n + 1) as usize);
    table.push(BigRational::one());
    for _ in 1..=n {
        // INVARIANT: table is never empty (initialized above)
        let prev = table.last().unwrap_or(&BigRational::one()).clone();
        table.push(prev * c);
    }
    table
}

/// Enumerate all compositions of n into k non-negative parts.
/// Calls f with each composition.
fn enumerate_compositions<F: FnMut(&[u32])>(n: u32, k: usize, buf: &mut [u32], f: &mut F) {
    debug_assert_eq!(buf.len(), k);
    enumerate_compositions_rec(n, k, 0, buf, f);
}

fn enumerate_compositions_rec<F: FnMut(&[u32])>(
    remaining: u32,
    total_k: usize,
    pos: usize,
    buf: &mut [u32],
    f: &mut F,
) {
    if pos == total_k - 1 {
        // Last position: must use all remaining
        buf[pos] = remaining;
        f(buf);
        return;
    }

    for val in 0..=remaining {
        buf[pos] = val;
        enumerate_compositions_rec(remaining - val, total_k, pos + 1, buf, f);
    }
}

/// Emit polynomial as Expr::Add tree from accumulated terms.
/// Uses balanced tree construction for better performance.
fn emit_polynomial_from_terms(
    ctx: &mut Context,
    var_ids: &[ExprId],
    terms: Vec<(MonoKey, BigRational)>,
) -> ExprId {
    if terms.is_empty() {
        return ctx.num(0);
    }

    // Convert each term to Expr
    let mut exprs: Vec<ExprId> = terms
        .into_iter()
        .map(|(mono, coeff)| emit_term(ctx, var_ids, &mono, coeff))
        .collect();

    // Build balanced Add tree
    while exprs.len() > 1 {
        let mut next = Vec::with_capacity(exprs.len().div_ceil(2));
        for chunk in exprs.chunks(2) {
            if chunk.len() == 2 {
                next.push(ctx.add(Expr::Add(chunk[0], chunk[1])));
            } else {
                next.push(chunk[0]);
            }
        }
        exprs = next;
    }

    // INVARIANT: loop reduces exprs until exactly 1 element remains.
    // Graceful fallback to 0 if invariant is ever violated.
    exprs.pop().unwrap_or_else(|| ctx.num(0))
}

/// Emit a single term: coeff * x1^e1 * x2^e2 * ...
fn emit_term(ctx: &mut Context, var_ids: &[ExprId], mono: &MonoKey, coeff: BigRational) -> ExprId {
    // Build monomial part
    let mut factors: Vec<ExprId> = Vec::new();

    for (i, &exp) in mono.0.iter().enumerate() {
        if exp == 0 {
            continue;
        }
        let var = var_ids[i];
        let factor = if exp == 1 {
            var
        } else {
            let e = ctx.num(exp as i64);
            ctx.add(Expr::Pow(var, e))
        };
        factors.push(factor);
    }

    // Build result
    let monomial = if factors.is_empty() {
        None
    } else {
        Some(build_balanced_mul(ctx, factors))
    };

    // Combine coefficient and monomial
    if coeff.is_one() {
        monomial.unwrap_or_else(|| ctx.num(1))
    } else if coeff == -BigRational::one() {
        let inner = monomial.unwrap_or_else(|| ctx.num(1));
        ctx.add(Expr::Neg(inner))
    } else {
        let coeff_expr = ctx.add(Expr::Number(coeff.clone()));
        match monomial {
            Some(m) => ctx.add(Expr::Mul(coeff_expr, m)),
            None => coeff_expr,
        }
    }
}

/// Build balanced Mul tree from factors.
fn build_balanced_mul(ctx: &mut Context, factors: Vec<ExprId>) -> ExprId {
    match factors.len() {
        0 => ctx.num(1),
        _ => ctx.build_balanced_mul(&factors),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_extract_linear_terms_simple() {
        let mut ctx = Context::new();
        let expr = parse("1 + 2*x + 3*y", &mut ctx).unwrap();
        let terms = extract_linear_terms(&ctx, expr).unwrap();
        assert_eq!(terms.len(), 3);
    }

    #[test]
    fn test_extract_linear_terms_negative() {
        let mut ctx = Context::new();
        let expr = parse("1 - 2*x", &mut ctx).unwrap();
        let terms = extract_linear_terms(&ctx, expr);
        assert!(terms.is_some());
    }

    #[test]
    fn test_multinomial_term_count() {
        // C(7+8-1, 8-1) = C(14,7) = 3432
        let count = multinomial_term_count(7, 8, 100_000).unwrap();
        assert_eq!(count, 3432);
    }

    #[test]
    fn test_factorial_table() {
        let fact = factorial_table(5);
        assert_eq!(fact[0], BigInt::from(1));
        assert_eq!(fact[5], BigInt::from(120));
    }

    #[test]
    fn test_multinomial_coeff() {
        let fact = factorial_table(4);
        // 4! / (2! * 1! * 1!) = 24 / 2 = 12
        let coeff = multinomial_coeff(&fact, 4, &[2, 1, 1]);
        assert_eq!(coeff, BigInt::from(12));
    }

    #[test]
    fn test_expand_binomial() {
        let mut ctx = Context::new();
        let expr = parse("(1 + x)^3", &mut ctx).unwrap();

        if let Expr::Pow(base, exp) = ctx.get(expr).clone() {
            let budget = MultinomialExpandBudget::default();
            let result = try_expand_multinomial_direct(&mut ctx, base, exp, &budget);
            assert!(result.is_some());
        }
    }

    #[test]
    fn test_expand_trinomial() {
        let mut ctx = Context::new();
        let expr = parse("(1 + 2*x + 3*y)^2", &mut ctx).unwrap();

        if let Expr::Pow(base, exp) = ctx.get(expr).clone() {
            let budget = MultinomialExpandBudget::default();
            let result = try_expand_multinomial_direct(&mut ctx, base, exp, &budget);
            assert!(result.is_some());
        }
    }
}
