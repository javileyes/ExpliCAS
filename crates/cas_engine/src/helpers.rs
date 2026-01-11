//! # Helpers Module
//!
//! This module provides shared utility functions for expression manipulation
//! and pattern matching. These functions are used across multiple rule files
//! to avoid code duplication.
//!
//! ## Categories
//!
//! - **Expression Predicates**: `is_one`, `is_zero`, `is_negative`, `is_half`
//! - **Value Extraction**: `get_integer`, `get_parts`, `get_variant_name`
//! - **Flattening**: `flatten_add`, `flatten_add_sub_chain`, `flatten_mul`, `flatten_mul_chain`
//! - **Trigonometry**: `is_trig_pow`, `get_trig_arg`, `extract_double_angle_arg`
//! - **Pi Helpers**: `is_pi`, `is_pi_over_n`, `build_pi_over_n`
//! - **Roots**: `get_square_root`

use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Signed, ToPrimitive};

// =============================================================================
// Expression Destructuring Helpers (Zero-Clone Pattern)
// =============================================================================
//
// These helpers extract child ExprIds without cloning the Expr enum.
// Use them instead of `ctx.get(id).clone()` to avoid unnecessary allocations.
//
// Pattern: Extract IDs first (in a short scope), then mutate ctx.

/// Destruct Add(l, r) -> Some((l, r)), else None
#[inline]
pub fn as_add(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Add(l, r) => Some((*l, *r)),
        _ => None,
    }
}

/// Destruct Sub(l, r) -> Some((l, r)), else None
#[inline]
pub fn as_sub(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Sub(l, r) => Some((*l, *r)),
        _ => None,
    }
}

/// Destruct Mul(l, r) -> Some((l, r)), else None
#[inline]
pub fn as_mul(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Mul(l, r) => Some((*l, *r)),
        _ => None,
    }
}

/// Destruct Div(l, r) -> Some((l, r)), else None
#[inline]
pub fn as_div(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Div(l, r) => Some((*l, *r)),
        _ => None,
    }
}

/// Destruct Pow(base, exp) -> Some((base, exp)), else None
#[inline]
pub fn as_pow(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Pow(b, e) => Some((*b, *e)),
        _ => None,
    }
}

/// Destruct Neg(inner) -> Some(inner), else None
#[inline]
pub fn as_neg(ctx: &Context, id: ExprId) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Neg(inner) => Some(*inner),
        _ => None,
    }
}

/// Extract function name and args without cloning String or Vec.
/// Returns references with the Context's lifetime.
#[inline]
pub fn fn_name_args(ctx: &Context, id: ExprId) -> Option<(&str, &[ExprId])> {
    match ctx.get(id) {
        Expr::Function(name, args) => Some((name.as_str(), args.as_slice())),
        _ => None,
    }
}

/// Check if expression matches a 1-arg function with the given name.
/// Returns the argument ExprId if matched.
#[inline]
pub fn as_fn1(ctx: &Context, id: ExprId, name: &str) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Function(fn_name, args) if fn_name == name && args.len() == 1 => Some(args[0]),
        _ => None,
    }
}

/// Check if expression matches a 2-arg function with the given name.
/// Returns the argument ExprIds if matched.
#[inline]
pub fn as_fn2(ctx: &Context, id: ExprId, name: &str) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Function(fn_name, args) if fn_name == name && args.len() == 2 => {
            Some((args[0], args[1]))
        }
        _ => None,
    }
}

/// Check if expression is a trigonometric function raised to a specific power.
///
/// # Example
/// ```ignore
/// // Matches sin(x)^2
/// is_trig_pow(ctx, expr, "sin", 2)
/// ```
pub fn is_trig_pow(context: &Context, expr: ExprId, name: &str, power: i64) -> bool {
    if let Expr::Pow(base, exp) = context.get(expr) {
        if let Expr::Number(n) = context.get(*exp) {
            if n.is_integer() && n.to_integer() == power.into() {
                if let Expr::Function(func_name, args) = context.get(*base) {
                    return func_name == name && args.len() == 1;
                }
            }
        }
    }
    false
}

/// Extract the argument from a trigonometric power expression.
///
/// For an expression like `sin(x)^2`, returns `Some(x)`.
pub fn get_trig_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, _) = context.get(expr) {
        if let Expr::Function(_, args) = context.get(*base) {
            if args.len() == 1 {
                return Some(args[0]);
            }
        }
    }
    None
}

pub fn get_square_root(context: &mut Context, expr: ExprId) -> Option<ExprId> {
    // We need to clone the expression to avoid borrowing issues if we need to inspect it deeply
    // But context.get returns reference.
    // We can't hold reference to context while mutating it.
    // So we should extract necessary data first.

    let expr_data = context.get(expr).clone();

    match expr_data {
        Expr::Pow(b, e) => {
            if let Expr::Number(n) = context.get(e) {
                if n.is_integer() {
                    let val = n.to_integer();
                    if &val % 2 == 0.into() {
                        let two = num_bigint::BigInt::from(2);
                        let new_exp_val = (val / two).to_i64()?;
                        let new_exp = context.num(new_exp_val);

                        // If new_exp is 1, simplify to b
                        if let Expr::Number(ne) = context.get(new_exp) {
                            if ne.is_one() {
                                return Some(b);
                            }
                        }
                        return Some(context.add(Expr::Pow(b, new_exp)));
                    }
                }
            }
            None
        }
        Expr::Number(n) => {
            if n.is_integer() && !n.is_negative() {
                let val = n.to_integer();
                let sqrt = val.sqrt();
                if &sqrt * &sqrt == val {
                    return Some(context.num(sqrt.to_i64()?));
                }
            }
            None
        }
        _ => None,
    }
}

pub fn extract_double_angle_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Mul(lhs, rhs) = context.get(expr) {
        if let Expr::Number(n) = context.get(*lhs) {
            if n.is_integer() && n.to_integer() == 2.into() {
                return Some(*rhs);
            }
        }
        if let Expr::Number(n) = context.get(*rhs) {
            if n.is_integer() && n.to_integer() == 2.into() {
                return Some(*lhs);
            }
        }
    }
    None
}

/// Extract inner variable from 3*x pattern (for triple angle identities).
/// Matches both Mul(3, x) and Mul(x, 3).
pub fn extract_triple_angle_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Mul(lhs, rhs) = context.get(expr) {
        if let Expr::Number(n) = context.get(*lhs) {
            if n.is_integer() && n.to_integer() == 3.into() {
                return Some(*rhs);
            }
        }
        if let Expr::Number(n) = context.get(*rhs) {
            if n.is_integer() && n.to_integer() == 3.into() {
                return Some(*lhs);
            }
        }
    }
    None
}

/// Flatten an Add chain into a list of terms (simple version).
///
/// This only handles `Add` nodes. For handling `Sub` and `Neg`, use
/// `flatten_add_sub_chain` instead.
pub fn flatten_add(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            flatten_add(ctx, *l, terms);
            flatten_add(ctx, *r, terms);
        }
        _ => terms.push(expr),
    }
}

/// Flatten an Add/Sub chain into a list of terms, converting subtractions to Neg.
/// This is used by collect and grouping modules for like-term collection.
///
/// Unlike `flatten_add`, this handles:
/// - `Add(a, b)` → [a, b]
/// - `Sub(a, b)` → [a, Neg(b)]
/// - `Neg(Neg(x))` → [x]
pub fn flatten_add_sub_chain(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    flatten_add_sub_recursive(ctx, expr, &mut terms, false);
    terms
}

fn flatten_add_sub_recursive(
    ctx: &mut Context,
    expr: ExprId,
    terms: &mut Vec<ExprId>,
    negate: bool,
) {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Add(l, r) => {
            flatten_add_sub_recursive(ctx, l, terms, negate);
            flatten_add_sub_recursive(ctx, r, terms, negate);
        }
        Expr::Sub(l, r) => {
            flatten_add_sub_recursive(ctx, l, terms, negate);
            flatten_add_sub_recursive(ctx, r, terms, !negate);
        }
        Expr::Neg(inner) => {
            // Handle double negation: Neg(Neg(x)) -> x
            flatten_add_sub_recursive(ctx, inner, terms, !negate);
        }
        _ => {
            if negate {
                terms.push(ctx.add(Expr::Neg(expr)));
            } else {
                terms.push(expr);
            }
        }
    }
}

/// Flatten a Mul chain into a list of factors (simple version).
///
/// This only handles `Mul` nodes. For handling `Neg` as `-1 * expr`,
/// use `flatten_mul_chain` instead.
pub fn flatten_mul(ctx: &Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            flatten_mul(ctx, *l, factors);
            flatten_mul(ctx, *r, factors);
        }
        _ => factors.push(expr),
    }
}

/// Flatten a Mul chain into a list of factors, handling Neg as -1 multiplication.
/// Returns Vec<ExprId> where Neg(e) is converted to [num(-1), ...factors of e...]
pub fn flatten_mul_chain(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    flatten_mul_recursive(ctx, expr, &mut factors);
    factors
}

fn flatten_mul_recursive(ctx: &mut Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Mul(l, r) => {
            flatten_mul_recursive(ctx, l, factors);
            flatten_mul_recursive(ctx, r, factors);
        }
        Expr::Neg(e) => {
            // Treat Neg(e) as -1 * e
            let neg_one = ctx.num(-1);
            factors.push(neg_one);
            flatten_mul_recursive(ctx, e, factors);
        }
        _ => {
            factors.push(expr);
        }
    }
}

pub fn get_parts(context: &mut Context, e: ExprId) -> (num_rational::BigRational, ExprId) {
    match context.get(e) {
        Expr::Mul(a, b) => {
            if let Expr::Number(n) = context.get(*a) {
                (n.clone(), *b)
            } else if let Expr::Number(n) = context.get(*b) {
                (n.clone(), *a)
            } else {
                (num_rational::BigRational::one(), e)
            }
        }
        Expr::Number(n) => (n.clone(), context.num(1)), // Treat constant as c * 1
        _ => (num_rational::BigRational::one(), e),
    }
}

// ========== Pi Helpers ==========

/// Check if expression equals π/n for a given denominator (handles both Div and Mul forms)
pub fn is_pi_over_n(ctx: &Context, expr: ExprId, denom: i32) -> bool {
    // Handle Div form: pi/n
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Constant(c) = ctx.get(*num) {
            if matches!(c, cas_ast::Constant::Pi) {
                if let Expr::Number(n) = ctx.get(*den) {
                    return *n == num_rational::BigRational::from_integer(denom.into());
                }
            }
        }
    }

    // Handle Mul form: (1/n) * pi
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let (num_part, const_part) = if let Expr::Constant(_) = ctx.get(*l) {
            (*r, *l)
        } else if let Expr::Constant(_) = ctx.get(*r) {
            (*l, *r)
        } else {
            return false;
        };

        if let Expr::Constant(c) = ctx.get(const_part) {
            if matches!(c, cas_ast::Constant::Pi) {
                if let Expr::Number(n) = ctx.get(num_part) {
                    return *n == num_rational::BigRational::new(1.into(), denom.into());
                }
            }
        }
    }

    false
}

/// Check if expression is exactly π
pub fn is_pi(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Constant(cas_ast::Constant::Pi))
}

/// Check if expression equals a specific numeric value
pub fn is_numeric_value(ctx: &Context, expr: ExprId, numer: i32, denom: i32) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n == num_rational::BigRational::new(numer.into(), denom.into())
    } else {
        false
    }
}

/// Build π/n expression
pub fn build_pi_over_n(ctx: &mut Context, denom: i64) -> ExprId {
    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
    let d = ctx.num(denom);
    ctx.add(Expr::Div(pi, d))
}

/// Check if expression equals 1/2
pub fn is_half(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n.numer() == 1.into() && *n.denom() == 2.into()
    } else {
        false
    }
}

// ========== Common Expression Predicates ==========
// These functions were previously duplicated across multiple files.
// Now consolidated here for consistency and maintainability.

/// Check if expression is the number 1
pub fn is_one(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        n.is_one()
    } else {
        false
    }
}

/// Check if expression is the number 0
pub fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        num_traits::Zero::is_zero(n)
    } else {
        false
    }
}

/// Check if expression is a negative number
pub fn is_negative(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        n.is_negative()
    } else {
        matches!(ctx.get(expr), Expr::Neg(_))
    }
}

/// Attempt to prove whether an expression is non-zero.
///
/// This is used by domain-aware simplification to gate operations like
/// `x/x → 1` which require the denominator to be non-zero.
///
/// Returns:
/// - `Proof::Proven` for expressions **provably** non-zero (2, π, -3, etc.)
/// - `Proof::Disproven` for expressions **provably** zero (0, 0/1, etc.)
/// - `Proof::Unknown` for variables, functions, and anything uncertain
///
/// This is intentionally conservative. `Unknown` means "we cannot prove it",
/// not "it might be zero".
///
/// # Examples
///
/// ```ignore
/// prove_nonzero(ctx, ctx.num(2))      // Proof::Proven
/// prove_nonzero(ctx, ctx.num(0))      // Proof::Disproven
/// prove_nonzero(ctx, ctx.var("x"))    // Proof::Unknown
/// prove_nonzero(ctx, ctx.pi())        // Proof::Proven
/// ```
pub fn prove_nonzero(ctx: &Context, expr: ExprId) -> crate::domain::Proof {
    use crate::domain::Proof;
    use num_traits::Zero;

    match ctx.get(expr) {
        // Numbers: check if zero
        Expr::Number(n) => {
            if n.is_zero() {
                Proof::Disproven
            } else {
                Proof::Proven
            }
        }

        // Constants: π, e, i are non-zero
        Expr::Constant(c) => {
            if matches!(
                c,
                cas_ast::Constant::Pi | cas_ast::Constant::E | cas_ast::Constant::I
            ) {
                Proof::Proven
            } else {
                Proof::Unknown
            }
        }

        // Neg: -a ≠ 0 iff a ≠ 0
        Expr::Neg(a) => prove_nonzero(ctx, *a),

        // Mul: a*b ≠ 0 iff a ≠ 0 AND b ≠ 0
        Expr::Mul(a, b) => {
            let proof_a = prove_nonzero(ctx, *a);
            let proof_b = prove_nonzero(ctx, *b);

            match (proof_a, proof_b) {
                (Proof::Disproven, _) | (_, Proof::Disproven) => Proof::Disproven,
                (Proof::Proven, Proof::Proven) => Proof::Proven,
                _ => Proof::Unknown,
            }
        }

        // Pow with positive integer exponent: a^n ≠ 0 iff a ≠ 0
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                // Only for positive exponents
                if n.is_integer() && n > &num_rational::BigRational::zero() {
                    return prove_nonzero(ctx, *base);
                }
            }
            Proof::Unknown
        }

        // Div: a/b ≠ 0 iff a ≠ 0 (assuming b ≠ 0 for the expression to be defined)
        Expr::Div(a, b) => {
            let proof_a = prove_nonzero(ctx, *a);
            let proof_b = prove_nonzero(ctx, *b);

            match (proof_a, proof_b) {
                // If numerator is 0, whole thing is 0
                (Proof::Disproven, _) => Proof::Disproven,
                // If a ≠ 0 and b ≠ 0, then a/b ≠ 0
                (Proof::Proven, Proof::Proven) => Proof::Proven,
                _ => Proof::Unknown,
            }
        }

        // ln(x) or log(x): non-zero iff x ≠ 1 (and x > 0 for it to be defined)
        // We check if x is a numeric constant that is > 0 and ≠ 1
        Expr::Function(name, args) if (name == "ln" || name == "log") && args.len() == 1 => {
            match ctx.get(args[0]) {
                Expr::Number(n) => {
                    let one = num_rational::BigRational::one();
                    let zero = num_rational::BigRational::zero();
                    if *n > zero && *n != one {
                        Proof::Proven // ln(x) where x > 0 and x ≠ 1 means ln(x) ≠ 0
                    } else if *n == one {
                        Proof::Disproven // ln(1) = 0
                    } else {
                        Proof::Unknown // x ≤ 0, ln undefined
                    }
                }
                // Check for division of two positive numbers ≠ 1 (like 3/5)
                Expr::Div(num, denom) => {
                    match (ctx.get(*num), ctx.get(*denom)) {
                        (Expr::Number(n), Expr::Number(d)) => {
                            let zero = num_rational::BigRational::zero();
                            if *n > zero && *d > zero && n != d {
                                Proof::Proven // ln(a/b) where a,b > 0 and a ≠ b means ≠ 0
                            } else if n == d {
                                Proof::Disproven // ln(1) = 0
                            } else {
                                Proof::Unknown
                            }
                        }
                        _ => Proof::Unknown,
                    }
                }
                Expr::Constant(c) => {
                    if matches!(c, cas_ast::Constant::Pi | cas_ast::Constant::E) {
                        Proof::Proven // ln(π) ≠ 0, ln(e) = 1 ≠ 0
                    } else {
                        Proof::Unknown
                    }
                }
                _ => Proof::Unknown,
            }
        }

        // Variables, other functions: UNKNOWN (conservative)
        _ => Proof::Unknown,
    }
}

/// Check if an expression can be proven to be non-zero (convenience wrapper).
///
/// Returns `true` only for `Proof::Proven`. Use `prove_nonzero()` directly
/// for more fine-grained control.
pub fn can_prove_nonzero(ctx: &Context, expr: ExprId) -> bool {
    prove_nonzero(ctx, expr).is_proven()
}

/// Attempt to prove whether an expression is strictly positive (> 0).
///
/// This is used by domain-aware simplification to gate operations like
/// `log(x*y) → log(x) + log(y)` which require both operands to be positive.
///
/// # Arguments
/// * `ctx` - Expression context
/// * `expr` - Expression to check
/// * `value_domain` - RealOnly or ComplexEnabled (affects what can be proven)
///
/// # ValueDomain semantics
/// * `RealOnly`: Symbols/variables are real by default. e^x > 0 for all x ∈ ℝ.
/// * `ComplexEnabled`: Symbols may be complex. Positivity only provable for numerics.
///
/// Returns:
/// - `Proof::Proven` for expressions **provably** > 0 (2, π, e, |x|^2, etc.)
/// - `Proof::Disproven` for expressions **provably** ≤ 0 (-3, 0, etc.)
/// - `Proof::Unknown` for variables, functions, and anything uncertain
///
/// # Examples
///
/// ```ignore
/// prove_positive(ctx, ctx.num(2), RealOnly)      // Proof::Proven
/// prove_positive(ctx, exp(x), RealOnly)          // Proof::Proven (x real)
/// prove_positive(ctx, exp(x), ComplexEnabled)   // Proof::Unknown (x may be complex)
/// ```
pub fn prove_positive(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> crate::domain::Proof {
    use crate::domain::Proof;
    use crate::semantics::ValueDomain;
    use num_traits::Zero;

    match ctx.get(expr) {
        // Numbers: check if > 0
        Expr::Number(n) => {
            if *n > num_rational::BigRational::zero() {
                Proof::Proven
            } else {
                Proof::Disproven // 0 or negative
            }
        }

        // Constants: π, e are positive; i is not (complex)
        Expr::Constant(c) => {
            if matches!(c, cas_ast::Constant::Pi | cas_ast::Constant::E) {
                Proof::Proven
            } else {
                Proof::Unknown // i, undefined, etc.
            }
        }

        // Mul: a*b > 0 if (a>0 AND b>0) OR (a<0 AND b<0)
        // V2.3: Also detect if either factor is exactly 0 (then product is 0, Disproven for >0)
        Expr::Mul(a, b) => {
            // Check if either factor is exactly 0
            if is_zero(ctx, *a) || is_zero(ctx, *b) {
                return Proof::Disproven; // 0 * anything = 0, which is not > 0
            }

            let proof_a = prove_positive(ctx, *a, value_domain);
            let proof_b = prove_positive(ctx, *b, value_domain);

            match (proof_a, proof_b) {
                (Proof::Proven, Proof::Proven) => Proof::Proven,
                // If either is ≤ 0, we can't easily conclude (could be positive if both negative)
                (Proof::Disproven, _) | (_, Proof::Disproven) => Proof::Unknown,
                _ => Proof::Unknown,
            }
        }

        // Div: a/b > 0 if (a>0 AND b>0) OR (a<0 AND b<0)
        Expr::Div(a, b) => {
            let proof_a = prove_positive(ctx, *a, value_domain);
            let proof_b = prove_positive(ctx, *b, value_domain);

            match (proof_a, proof_b) {
                (Proof::Proven, Proof::Proven) => Proof::Proven,
                (Proof::Disproven, _) | (_, Proof::Disproven) => Proof::Unknown,
                _ => Proof::Unknown,
            }
        }

        // Pow: base^exp
        // - RealOnly: if base > 0, then base^exp > 0 (for any real exp)
        // - ComplexEnabled: only if exp is a real numeric AND base > 0
        Expr::Pow(base, exp) => {
            let base_positive = prove_positive(ctx, *base, value_domain);

            match value_domain {
                ValueDomain::RealOnly => {
                    // In reals: positive^(anything real) = positive
                    if base_positive == Proof::Proven {
                        return Proof::Proven;
                    }
                }
                ValueDomain::ComplexEnabled => {
                    // In complex: only safe if exponent is a real numeric
                    let exp_is_real_numeric = matches!(ctx.get(*exp), Expr::Number(_));
                    if base_positive == Proof::Proven && exp_is_real_numeric {
                        return Proof::Proven;
                    }
                }
            }

            // Check for even power (makes result positive if base ≠ 0)
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() {
                    let int_val = n.to_integer();
                    let two: num_bigint::BigInt = 2.into();
                    if &int_val % &two == 0.into() {
                        // a^(even) > 0 if a ≠ 0
                        let base_nonzero = prove_nonzero(ctx, *base);
                        if base_nonzero == Proof::Proven {
                            return Proof::Proven;
                        }
                    }
                }
            }
            Proof::Unknown
        }

        // abs(x): always ≥ 0, but only > 0 if x ≠ 0
        Expr::Function(name, args) if name == "abs" && args.len() == 1 => {
            let inner_nonzero = prove_nonzero(ctx, args[0]);
            if inner_nonzero == Proof::Proven {
                Proof::Proven
            } else if inner_nonzero == Proof::Disproven {
                Proof::Disproven // |0| = 0
            } else {
                Proof::Unknown
            }
        }

        // exp(x) > 0 for all x ∈ ℝ, but NOT for complex x
        // RealOnly: symbols are real, so exp(symbol) > 0
        // ComplexEnabled: only exp(literal) is provably positive
        Expr::Function(name, args) if name == "exp" && args.len() == 1 => {
            match value_domain {
                ValueDomain::RealOnly => {
                    // In RealOnly: e^x > 0 for ALL x (x is real by contract)
                    Proof::Proven
                }
                ValueDomain::ComplexEnabled => {
                    // In ComplexEnabled: only exp(numeric literal) is provably positive
                    match ctx.get(args[0]) {
                        Expr::Number(_)
                        | Expr::Constant(cas_ast::Constant::Pi)
                        | Expr::Constant(cas_ast::Constant::E) => Proof::Proven,
                        _ => Proof::Unknown,
                    }
                }
            }
        }

        // sqrt(x) with x > 0 gives positive result
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
            prove_positive(ctx, args[0], value_domain)
        }

        // Neg: -x > 0 iff x < 0
        // If x is proven positive (> 0), then -x is proven negative (< 0), so Disproven
        // If x is proven negative (< 0), then -x is proven positive (> 0), so Proven
        Expr::Neg(inner) => {
            let inner_proof = prove_positive(ctx, *inner, value_domain);
            match inner_proof {
                Proof::Proven => Proof::Disproven, // -(positive) = negative
                Proof::Disproven => {
                    // Inner is ≤ 0. If inner is strictly < 0, -inner > 0
                    // But we can't distinguish < 0 from = 0 here, so Unknown for now
                    // However, for literals like Neg(Number(5)), inner is 5 > 0 → Proven
                    // So this case handles Neg(negative_number) = positive
                    // Check if inner is a negative number
                    if let Expr::Number(n) = ctx.get(*inner) {
                        if n.is_negative() {
                            return Proof::Proven; // -(-n) = n > 0
                        }
                    }
                    Proof::Unknown
                }
                _ => Proof::Unknown,
            }
        }

        // Variables, other functions: UNKNOWN (conservative)
        _ => Proof::Unknown,
    }
}

/// Check if an expression can be proven to be positive (convenience wrapper).
///
/// Returns `true` only for `Proof::Proven`. Use `prove_positive()` directly
/// for more fine-grained control.
pub fn can_prove_positive(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> bool {
    prove_positive(ctx, expr, value_domain).is_proven()
}

/// Attempt to prove whether an expression is non-negative (≥ 0).
///
/// This is used by domain-aware simplification to gate operations like
/// `sqrt(x)² → x` which require x ≥ 0 in reals.
///
/// IMPORTANT: This is different from `prove_positive`:
/// - `prove_positive`: proves x > 0 (strictly positive)
/// - `prove_nonnegative`: proves x ≥ 0 (non-negative, includes zero)
///
/// # Returns
/// - `Proof::Proven` for expressions **provably** ≥ 0 (0, 2, π, sqrt(x), |x|, x², etc.)
/// - `Proof::Disproven` for expressions **provably** < 0 (-3, etc.)
/// - `Proof::Unknown` for variables, functions, and anything uncertain
pub fn prove_nonnegative(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> crate::domain::Proof {
    use crate::domain::Proof;
    use crate::semantics::ValueDomain;
    use num_traits::Zero;

    match ctx.get(expr) {
        // Numbers: check if ≥ 0
        Expr::Number(n) => {
            if *n >= num_rational::BigRational::zero() {
                Proof::Proven
            } else {
                Proof::Disproven // negative
            }
        }

        // Constants: π, e are positive (hence non-negative); i is not (complex)
        Expr::Constant(c) => {
            if matches!(c, cas_ast::Constant::Pi | cas_ast::Constant::E) {
                Proof::Proven
            } else {
                Proof::Unknown // i, undefined, etc.
            }
        }

        // Pow: base^exp
        // Even powers are always non-negative: x² ≥ 0
        Expr::Pow(base, exp) => {
            // Check for even power (makes result non-negative for any real base)
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() {
                    let int_val = n.to_integer();
                    let two: num_bigint::BigInt = 2.into();
                    if &int_val % &two == 0.into() && int_val > 0.into() {
                        // a^(positive even) ≥ 0 always in reals
                        return Proof::Proven;
                    }
                }
            }

            // Positive base with any exponent in reals is positive (hence non-negative)
            if value_domain == ValueDomain::RealOnly {
                let base_positive = prove_positive(ctx, *base, value_domain);
                if base_positive == Proof::Proven {
                    return Proof::Proven;
                }
            }

            Proof::Unknown
        }

        // abs(x): always ≥ 0
        Expr::Function(name, args) if name == "abs" && args.len() == 1 => Proof::Proven,

        // sqrt(x): if defined, result is ≥ 0 (by principal root convention)
        // But we can't prove sqrt is defined without proving arg ≥ 0 (circular)
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
            // If the arg is provably non-negative, sqrt(arg) ≥ 0
            prove_nonnegative(ctx, args[0], value_domain)
        }

        // exp(x) > 0 for all real x, hence ≥ 0
        Expr::Function(name, args) if name == "exp" && args.len() == 1 => {
            match value_domain {
                ValueDomain::RealOnly => Proof::Proven, // e^x > 0 for all real x
                ValueDomain::ComplexEnabled => {
                    // Only exp(numeric literal) is provably positive
                    match ctx.get(args[0]) {
                        Expr::Number(_)
                        | Expr::Constant(cas_ast::Constant::Pi)
                        | Expr::Constant(cas_ast::Constant::E) => Proof::Proven,
                        _ => Proof::Unknown,
                    }
                }
            }
        }

        // Mul of two non-negative numbers is non-negative
        Expr::Mul(a, b) => {
            let proof_a = prove_nonnegative(ctx, *a, value_domain);
            let proof_b = prove_nonnegative(ctx, *b, value_domain);

            match (proof_a, proof_b) {
                (Proof::Proven, Proof::Proven) => Proof::Proven,
                _ => Proof::Unknown, // Can't easily prove otherwise
            }
        }

        // Neg: -x ≥ 0 iff x ≤ 0 - check if x is ≤ 0
        Expr::Neg(inner) => {
            // If inner is provably ≤ 0 (disproven as non-negative), then -inner ≥ 0
            let inner_proof = prove_nonnegative(ctx, *inner, value_domain);
            match inner_proof {
                Proof::Disproven => Proof::Proven, // -(-3) = 3 ≥ 0
                _ => Proof::Unknown,
            }
        }

        // Variables, other functions: UNKNOWN (conservative)
        _ => Proof::Unknown,
    }
}

/// Prove non-negative with implicit domain support.
///
/// This extends `prove_nonnegative` to also consider conditions that are
/// implicitly required by the expression structure (e.g., `sqrt(x)` implies `x ≥ 0`).
///
/// Returns `Proof::ProvenImplicit` if:
/// 1. The base proof is `Unknown`
/// 2. The expression has an implicit non-negative constraint
/// 3. The witness for that constraint survives in the output
///
/// This enables simplifications like `sqrt(x)² → x` within expressions
/// that still contain `sqrt(x)` elsewhere, without requiring explicit assumptions.
pub fn prove_nonnegative_with_implicit(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
    implicit_domain: &crate::implicit_domain::ImplicitDomain,
    output: ExprId,
) -> crate::domain::Proof {
    use crate::domain::Proof;
    use crate::implicit_domain::WitnessKind;

    // First try normal proof
    let base_proof = prove_nonnegative(ctx, expr, value_domain);

    match base_proof {
        Proof::Proven | Proof::ProvenImplicit | Proof::Disproven => base_proof,
        Proof::Unknown => {
            // Check if implicit domain contains NonNegative(expr)
            if implicit_domain.contains_nonnegative(expr) {
                // Check if witness survives in output
                if crate::implicit_domain::witness_survives(ctx, expr, output, WitnessKind::Sqrt) {
                    return Proof::ProvenImplicit;
                }
            }
            Proof::Unknown
        }
    }
}

/// V2.0: Prove positivity with guard environment.
///
/// Like `prove_positive`, but first checks if the expression's positivity
/// is asserted in the guard set. This allows conditional branches to
/// treat guarded conditions as proven facts.
///
/// # Arguments
/// * `ctx` - Expression context
/// * `expr` - Expression to check
/// * `value_domain` - RealOnly or ComplexEnabled
/// * `guards` - Set of conditions to treat as proven
///
/// # Example
/// ```ignore
/// let guard = ConditionSet::single(ConditionPredicate::Positive(x));
/// // Now prove_positive_with_guards(ctx, x, RealOnly, &guard) returns Proven
/// ```
pub fn prove_positive_with_guards(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
    guards: &cas_ast::ConditionSet,
) -> crate::domain::Proof {
    use crate::domain::Proof;
    use cas_ast::ConditionPredicate;

    // First check if this expression is guarded as Positive
    for pred in guards.predicates() {
        match pred {
            ConditionPredicate::Positive(e) if *e == expr => return Proof::Proven,
            ConditionPredicate::NonNegative(e) if *e == expr => {
                // NonNegative doesn't imply strictly positive, but if we have
                // NonNegative AND NonZero, then we can prove Positive
                if guards
                    .predicates()
                    .iter()
                    .any(|p| matches!(p, ConditionPredicate::NonZero(z) if *z == expr))
                {
                    return Proof::Proven;
                }
            }
            _ => {}
        }
    }

    // Fall back to normal proof
    prove_positive(ctx, expr, value_domain)
}

/// V2.0: Prove non-zero with guard environment.
pub fn prove_nonzero_with_guards(
    ctx: &Context,
    expr: ExprId,
    guards: &cas_ast::ConditionSet,
) -> crate::domain::Proof {
    use crate::domain::Proof;
    use cas_ast::ConditionPredicate;

    // Check if this expression is guarded as NonZero or Positive
    for pred in guards.predicates() {
        match pred {
            ConditionPredicate::NonZero(e) if *e == expr => return Proof::Proven,
            ConditionPredicate::Positive(e) if *e == expr => return Proof::Proven, // x > 0 implies x ≠ 0
            _ => {}
        }
    }

    // Fall back to normal proof
    prove_nonzero(ctx, expr)
}

// ========== Solver Domain Helpers ==========

/// Decision result for `can_take_ln_real`.
///
/// Used by the solver to determine if ln(arg) is valid in RealOnly mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LnDecision {
    /// Argument is provably positive - ln() is safe with no assumption.
    Safe,
    /// Argument positivity is unknown but allowed under assumption (Assume mode only).
    /// Contains the assumption message to be emitted.
    AssumePositive,
}

/// Check if ln(arg) is valid in RealOnly mode.
///
/// This is used by the solver to gate log operations on exponential equations.
///
/// # Arguments
/// * `ctx` - Expression context
/// * `arg` - The argument to ln()
/// * `mode` - The current DomainMode
/// * `value_domain` - RealOnly or ComplexEnabled
///
/// # Returns
/// * `Ok(LnDecision::Safe)` if arg is provably positive (no assumption needed)
/// * `Ok(LnDecision::AssumePositive)` if allowed with assumption (Assume mode only)
/// * `Err(reason)` if ln is invalid (arg ≤ 0 proven, or unknown in Strict/Generic)
///
/// # Examples
/// ```ignore
/// can_take_ln_real(ctx, ctx.num(2), DomainMode::Strict, RealOnly)   // Ok(Safe)
/// can_take_ln_real(ctx, ctx.num(-5), DomainMode::Strict, RealOnly)  // Err("argument ≤ 0")
/// can_take_ln_real(ctx, ctx.var("x"), DomainMode::Strict, RealOnly) // Err("cannot prove > 0")
/// can_take_ln_real(ctx, ctx.var("x"), DomainMode::Assume, RealOnly) // Ok(AssumePositive)
/// ```
pub fn can_take_ln_real(
    ctx: &Context,
    arg: ExprId,
    mode: crate::domain::DomainMode,
    value_domain: crate::semantics::ValueDomain,
) -> Result<LnDecision, &'static str> {
    use crate::domain::{DomainMode, Proof};

    let proof = prove_positive(ctx, arg, value_domain);

    match proof {
        Proof::Proven | Proof::ProvenImplicit => Ok(LnDecision::Safe),
        Proof::Disproven => Err("argument is ≤ 0"),
        Proof::Unknown => match mode {
            DomainMode::Strict | DomainMode::Generic => Err("cannot prove argument > 0 for ln()"),
            DomainMode::Assume => Ok(LnDecision::AssumePositive),
        },
    }
}

/// Try to extract an integer value from an expression.
///
/// Returns `None` if:
/// - Expression is not a Number
/// - Number is not an integer (has non-1 denominator)
/// - Integer value doesn't fit in `i64`
///
/// For BigInt extraction without i64 limitations, use `get_integer_exact`.
pub fn get_integer(ctx: &Context, expr: ExprId) -> Option<i64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            n.to_integer().to_i64()
        } else {
            None
        }
    } else {
        None
    }
}

/// Extract an integer value from an expression as BigInt.
///
/// Unlike `get_integer`, this:
/// - Returns the full BigInt without i64 truncation
/// - Also handles `Neg(e)` by recursively extracting and negating
///
/// Use this for number theory operations where large integers are expected.
pub fn get_integer_exact(ctx: &Context, expr: ExprId) -> Option<num_bigint::BigInt> {
    match ctx.get(expr) {
        Expr::Number(n) => {
            if n.is_integer() {
                Some(n.to_integer())
            } else {
                None
            }
        }
        Expr::Neg(e) => get_integer_exact(ctx, *e).map(|n| -n),
        _ => None,
    }
}

/// Get the variant name of an expression (for debugging/display)
pub fn get_variant_name(expr: &Expr) -> &'static str {
    match expr {
        Expr::Number(_) => "Number",
        Expr::Variable(_) => "Variable",
        Expr::Constant(_) => "Constant",
        Expr::Add(_, _) => "Add",
        Expr::Sub(_, _) => "Sub",
        Expr::Mul(_, _) => "Mul",
        Expr::Div(_, _) => "Div",
        Expr::Pow(_, _) => "Pow",
        Expr::Neg(_) => "Neg",
        Expr::Function(_, _) => "Function",
        Expr::Matrix { .. } => "Matrix",
        Expr::SessionRef(_) => "SessionRef",
    }
}

// ========== Normal Form Scoring ==========

/// Count total nodes in an expression tree
pub fn count_all_nodes(ctx: &Context, expr: ExprId) -> usize {
    let mut count = 0;
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        count += 1;
        match ctx.get(id) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(e) => stack.push(*e),
            Expr::Function(_, args) => stack.extend(args),
            Expr::Matrix { data, .. } => stack.extend(data),
            _ => {}
        }
    }
    count
}

/// Count nodes matching a predicate.
///
/// Wrapper calling canonical `cas_ast::traversal::count_nodes_matching`.
/// (See POLICY.md "Traversal Contract")
pub fn count_nodes_matching<F>(ctx: &Context, expr: ExprId, pred: F) -> usize
where
    F: FnMut(&Expr) -> bool,
{
    cas_ast::traversal::count_nodes_matching(ctx, expr, pred)
}

/// Score expression for normal form quality (lower is better).
/// Returns (divs_subs, total_nodes, mul_inversions) for lexicographic comparison.
///
/// Expressions with fewer Div/Sub nodes are preferred (C2 canonical form).
/// Ties are broken by total node count (simpler is better).
/// Final tie-breaker: fewer out-of-order adjacent pairs in Mul chains.
///
/// For performance-critical comparisons, use `compare_nf_score_lazy` instead.
pub fn nf_score(ctx: &Context, id: ExprId) -> (usize, usize, usize) {
    let divs_subs = count_nodes_matching(ctx, id, |e| matches!(e, Expr::Div(..) | Expr::Sub(..)));
    let total = count_all_nodes(ctx, id);
    let inversions = mul_unsorted_adjacent(ctx, id);
    (divs_subs, total, inversions)
}

/// First two components of nf_score: (divs_subs, total_nodes)
/// Uses single traversal for efficiency (counts both in one pass).
fn nf_score_base(ctx: &Context, id: ExprId) -> (usize, usize) {
    let mut divs_subs = 0;
    let mut total = 0;
    let mut stack = vec![id];

    while let Some(node_id) = stack.pop() {
        total += 1;

        match ctx.get(node_id) {
            Expr::Div(..) | Expr::Sub(..) => divs_subs += 1,
            _ => {}
        }

        // Push children
        match ctx.get(node_id) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args),
            Expr::Matrix { data, .. } => stack.extend(data),
            _ => {}
        }
    }

    (divs_subs, total)
}

/// Compare nf_score lazily: only computes mul_unsorted_adjacent if first two components tie.
/// Returns true if `after` is strictly better (lower) than `before`.
pub fn nf_score_after_is_better(ctx: &Context, before: ExprId, after: ExprId) -> bool {
    let before_base = nf_score_base(ctx, before);
    let after_base = nf_score_base(ctx, after);

    // Compare first two components
    if after_base < before_base {
        return true; // Clear improvement
    }
    if after_base > before_base {
        return false; // Worse
    }

    // Tie on (divs_subs, total) - need to compare mul_inversions
    let before_inv = mul_unsorted_adjacent(ctx, before);
    let after_inv = mul_unsorted_adjacent(ctx, after);
    after_inv < before_inv
}

/// Count out-of-order adjacent pairs in Mul chains (right-associative).
///
/// For a chain `a * (b * (c * d))` with factors `[a, b, c, d]`:
/// - Counts how many pairs (f[i], f[i+1]) have compare_expr(f[i], f[i+1]) == Greater
///
/// This metric allows canonicalizing rewrites that only reorder Mul factors.
pub fn mul_unsorted_adjacent(ctx: &Context, root: ExprId) -> usize {
    use crate::ordering::compare_expr;
    use std::cmp::Ordering;
    use std::collections::HashSet;

    // Collect all Mul nodes and identify which are right-children of other Muls
    let mut mul_nodes: HashSet<ExprId> = HashSet::new();
    let mut mul_right_children: HashSet<ExprId> = HashSet::new();

    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Mul(l, r) => {
                mul_nodes.insert(id);
                if matches!(ctx.get(*r), Expr::Mul(..)) {
                    mul_right_children.insert(*r);
                }
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args),
            Expr::Matrix { data, .. } => stack.extend(data),
            _ => {}
        }
    }

    // Heads are Mul nodes that are NOT the right child of another Mul
    let heads: Vec<_> = mul_nodes.difference(&mul_right_children).copied().collect();

    let mut inversions = 0;

    for head in heads {
        // Linearize factors by following right-assoc pattern: a*(b*(c*d)) -> [a,b,c,d]
        let mut factors = Vec::new();
        let mut current = head;

        loop {
            if let Expr::Mul(l, r) = ctx.get(current).clone() {
                factors.push(l);
                if matches!(ctx.get(r), Expr::Mul(..)) {
                    current = r;
                } else {
                    factors.push(r);
                    break;
                }
            } else {
                factors.push(current);
                break;
            }
        }

        // Count adjacent inversions
        for i in 0..factors.len().saturating_sub(1) {
            if compare_expr(ctx, factors[i], factors[i + 1]) == Ordering::Greater {
                inversions += 1;
            }
        }
    }

    inversions
}

// ========== Numeric Evaluation ==========

/// Default depth limit for numeric evaluation.
/// Prevents stack overflow on deeply nested expressions.
pub const DEFAULT_NUMERIC_EVAL_DEPTH: usize = 50;

/// Extract a rational constant from an expression, handling multiple representations.
/// Uses default depth limit (50) to prevent stack overflow.
///
/// Supports (all must be purely numeric - returns None if any variable/function present):
/// - `Number(n)` - direct rational
/// - `Div(a, b)` - fraction (recursive)
/// - `Neg(a)` - negation (recursive)
/// - `Mul(a, b)` - product (recursive)
/// - `Add(a, b)` - sum (recursive)
/// - `Sub(a, b)` - difference (recursive)
///
/// This is the canonical helper for numeric evaluation. Used by:
/// - `SemanticEqualityChecker::try_evaluate_numeric`
/// - `EvaluatePowerRule` for exponent matching
pub fn as_rational_const(ctx: &Context, expr: ExprId) -> Option<num_rational::BigRational> {
    as_rational_const_depth(ctx, expr, DEFAULT_NUMERIC_EVAL_DEPTH)
}

/// Extract a rational constant with explicit depth limit.
/// Returns None if depth is exhausted (prevents stack overflow on deep expressions).
pub fn as_rational_const_depth(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
) -> Option<num_rational::BigRational> {
    use num_traits::Zero;

    if depth == 0 {
        return None; // Depth budget exhausted
    }

    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),

        Expr::Div(num, den) => {
            let n = as_rational_const_depth(ctx, *num, depth - 1)?;
            let d = as_rational_const_depth(ctx, *den, depth - 1)?;
            if !d.is_zero() {
                Some(n / d)
            } else {
                None
            }
        }

        Expr::Neg(inner) => {
            let val = as_rational_const_depth(ctx, *inner, depth - 1)?;
            Some(-val)
        }

        Expr::Mul(l, r) => {
            let lv = as_rational_const_depth(ctx, *l, depth - 1)?;
            let rv = as_rational_const_depth(ctx, *r, depth - 1)?;
            Some(lv * rv)
        }

        Expr::Add(l, r) => {
            let lv = as_rational_const_depth(ctx, *l, depth - 1)?;
            let rv = as_rational_const_depth(ctx, *r, depth - 1)?;
            Some(lv + rv)
        }

        Expr::Sub(l, r) => {
            let lv = as_rational_const_depth(ctx, *l, depth - 1)?;
            let rv = as_rational_const_depth(ctx, *r, depth - 1)?;
            Some(lv - rv)
        }

        // Variables, Constants, Functions, Pow, Matrix -> not purely numeric
        _ => None,
    }
}

/// Check if an expression contains an integral (for auto-context detection).
///
/// Searches the expression tree for `integrate(...)` function calls.
/// Uses iterative traversal to avoid stack overflow on deep expressions.
pub fn contains_integral(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];

    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Function(name, _) if name == "integrate" || name == "int" => {
                return true;
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => {
                stack.push(*inner);
            }
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push(*arg);
                }
            }
            Expr::Matrix { data, .. } => {
                for elem in data {
                    stack.push(*elem);
                }
            }
            Expr::Div(num, den) => {
                stack.push(*num);
                stack.push(*den);
            }
            // Leaf nodes: nothing to push
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

/// Check if an expression contains the imaginary unit `i` anywhere.
/// Check if an expression contains the imaginary unit `i` or imaginary-producing expressions.
/// Detects: Constant::I, sqrt(-1), (-1)^(1/2), and similar patterns.
/// Uses iterative traversal to avoid stack overflow on deep expressions.
pub fn contains_i(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];

    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Constant(c) if *c == cas_ast::Constant::I => {
                return true;
            }
            // Check for sqrt(-1) pattern
            Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
                if is_negative_one(ctx, args[0]) {
                    return true;
                }
                // Still need to traverse the arg for nested i
                stack.push(args[0]);
            }
            // Check for (-1)^(1/2) pattern
            Expr::Pow(base, exp) => {
                if is_negative_one(ctx, *base) && is_one_half(ctx, *exp) {
                    return true;
                }
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => {
                stack.push(*inner);
            }
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push(*arg);
                }
            }
            Expr::Matrix { data, .. } => {
                for elem in data {
                    stack.push(*elem);
                }
            }
            Expr::Div(num, den) => {
                stack.push(*num);
                stack.push(*den);
            }
            // Leaf nodes: nothing to push
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

/// Check if an expression represents -1
fn is_negative_one(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == num_rational::BigRational::from_integer((-1).into()),
        Expr::Neg(inner) => {
            matches!(
                ctx.get(*inner),
                Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
            )
        }
        _ => false,
    }
}

/// Check if an expression represents 1/2
fn is_one_half(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == num_rational::BigRational::new(1.into(), 2.into()),
        Expr::Div(num, den) => {
            matches!((ctx.get(*num), ctx.get(*den)),
                (Expr::Number(n), Expr::Number(d))
                if *n == num_rational::BigRational::from_integer(1.into())
                && *d == num_rational::BigRational::from_integer(2.into())
            )
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_is_one() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let zero = ctx.num(0);
        let x = ctx.var("x");

        assert!(is_one(&ctx, one));
        assert!(!is_one(&ctx, two));
        assert!(!is_one(&ctx, zero));
        assert!(!is_one(&ctx, x));
    }

    #[test]
    fn test_is_zero() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let one = ctx.num(1);
        let x = ctx.var("x");

        assert!(is_zero(&ctx, zero));
        assert!(!is_zero(&ctx, one));
        assert!(!is_zero(&ctx, x));
    }

    #[test]
    fn test_is_negative() {
        let mut ctx = Context::new();
        let neg_one = ctx.num(-1);
        let one = ctx.num(1);
        let x = ctx.var("x");
        let neg_x = ctx.add(Expr::Neg(x));

        assert!(is_negative(&ctx, neg_one));
        assert!(!is_negative(&ctx, one));
        assert!(is_negative(&ctx, neg_x));
        assert!(!is_negative(&ctx, x));
    }

    #[test]
    fn test_get_integer() {
        let mut ctx = Context::new();
        let five = ctx.num(5);
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let x = ctx.var("x");

        assert_eq!(get_integer(&ctx, five), Some(5));
        assert_eq!(get_integer(&ctx, half), None); // Not an integer
        assert_eq!(get_integer(&ctx, x), None);
    }

    #[test]
    fn test_flatten_add() {
        let mut ctx = Context::new();
        let expr = parse("a + b + c", &mut ctx).unwrap();
        let mut terms = Vec::new();
        flatten_add(&ctx, expr, &mut terms);
        assert_eq!(terms.len(), 3);
    }

    #[test]
    fn test_flatten_add_sub_chain() {
        let mut ctx = Context::new();
        let expr = parse("a + b - c", &mut ctx).unwrap();
        let terms = flatten_add_sub_chain(&mut ctx, expr);
        // Should have 3 terms: a, b, Neg(c)
        assert_eq!(terms.len(), 3);
    }

    #[test]
    fn test_flatten_mul() {
        let mut ctx = Context::new();
        let expr = parse("a * b * c", &mut ctx).unwrap();
        let mut factors = Vec::new();
        flatten_mul(&ctx, expr, &mut factors);
        assert_eq!(factors.len(), 3);
    }

    #[test]
    fn test_flatten_mul_chain_with_neg() {
        let mut ctx = Context::new();
        let expr = parse("-a * b", &mut ctx).unwrap();
        let factors = flatten_mul_chain(&mut ctx, expr);
        // Should have factors including -1
        assert!(factors.len() >= 2);
    }

    #[test]
    fn test_get_variant_name() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);

        // Test with actual expressions from context
        assert_eq!(get_variant_name(ctx.get(a)), "Number");

        let x = ctx.var("x");
        assert_eq!(get_variant_name(ctx.get(x)), "Variable");

        let sum = ctx.add(Expr::Add(a, b));
        assert_eq!(get_variant_name(ctx.get(sum)), "Add");
    }

    #[test]
    fn test_is_pi_over_n() {
        let mut ctx = Context::new();
        let pi_over_2 = build_pi_over_n(&mut ctx, 2);
        let pi_over_4 = build_pi_over_n(&mut ctx, 4);

        assert!(is_pi_over_n(&ctx, pi_over_2, 2));
        assert!(!is_pi_over_n(&ctx, pi_over_2, 4));
        assert!(is_pi_over_n(&ctx, pi_over_4, 4));
    }

    #[test]
    fn test_is_half() {
        let mut ctx = Context::new();
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let one = ctx.num(1);

        assert!(is_half(&ctx, half));
        assert!(!is_half(&ctx, one));
    }

    #[test]
    fn test_is_pi() {
        let mut ctx = Context::new();
        let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let x = ctx.var("x");

        assert!(is_pi(&ctx, pi));
        assert!(!is_pi(&ctx, e));
        assert!(!is_pi(&ctx, x));
    }

    #[test]
    fn test_as_rational_const_number() {
        let mut ctx = Context::new();
        let half = ctx.rational(1, 2);
        let result = as_rational_const(&ctx, half);
        assert_eq!(
            result,
            Some(num_rational::BigRational::new(1.into(), 2.into()))
        );
    }

    #[test]
    fn test_as_rational_const_div() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let div = ctx.add(Expr::Div(one, two));
        let result = as_rational_const(&ctx, div);
        assert_eq!(
            result,
            Some(num_rational::BigRational::new(1.into(), 2.into()))
        );
    }

    #[test]
    fn test_as_rational_const_neg_div() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let div = ctx.add(Expr::Div(one, two));
        let neg = ctx.add(Expr::Neg(div));
        let result = as_rational_const(&ctx, neg);
        assert_eq!(
            result,
            Some(num_rational::BigRational::new((-1).into(), 2.into()))
        );
    }

    #[test]
    fn test_as_rational_const_variable_returns_none() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        assert!(as_rational_const(&ctx, x).is_none());
    }

    #[test]
    fn test_as_rational_const_mul_with_variable_returns_none() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        let mul = ctx.add(Expr::Mul(two, x));
        assert!(as_rational_const(&ctx, mul).is_none());
    }

    #[test]
    fn test_as_rational_const_depth_budget() {
        // Build a deeply nested expression: Div(1, Div(1, Div(1, ...Div(1, 2)...)))
        // Note: We can't use Neg(Neg(...)) because Context::add canonicalizes Neg(Neg(x)) -> x
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);

        // Start with 1/2, then nest 100 levels: 1/(1/(1/(...1/2...)))
        let mut expr = ctx.add(Expr::Div(one, two));
        for _ in 0..100 {
            let one_copy = ctx.num(1);
            expr = ctx.add(Expr::Div(one_copy, expr));
        }

        // With depth=50, should return None (budget exhausted)
        assert!(as_rational_const_depth(&ctx, expr, 50).is_none());

        // With depth=200, should succeed
        // 1/(1/(1/...(1/2)...)) with even nesting = 2, odd nesting = 1/2
        // 100 levels of nesting on 1/2 => result depends on parity
        let result = as_rational_const_depth(&ctx, expr, 200);
        assert!(result.is_some());
        // 100 nestings of 1/x on 1/2: alternates between 2 and 1/2
        // Even count (100) means result is 1/2
        let expected = num_rational::BigRational::new(1.into(), 2.into());
        assert_eq!(result.unwrap(), expected);
    }

    /// Test that add_raw preserves operand order while add() canonicalizes
    #[test]
    fn test_add_raw_preserves_mul_order() {
        let mut ctx = Context::new();
        let z = ctx.var("z"); // 'z' > 'a' in ordering
        let a = ctx.var("a");

        // With add(): z * a → a * z (swapped because 'z' > 'a')
        let mul_canonical = ctx.add(Expr::Mul(z, a));
        if let Expr::Mul(l, r) = ctx.get(mul_canonical) {
            // Should be swapped to canonical order: (a, z)
            assert_eq!(*l, a, "add() should swap to put 'a' first");
            assert_eq!(*r, z, "add() should swap to put 'z' second");
        } else {
            panic!("Expected Mul expression");
        }

        // With add_raw(): z * a → z * a (preserved order)
        let mul_raw = ctx.add_raw(Expr::Mul(z, a));
        if let Expr::Mul(l, r) = ctx.get(mul_raw) {
            // Should preserve original order: (z, a)
            assert_eq!(*l, z, "add_raw() should preserve 'z' first");
            assert_eq!(*r, a, "add_raw() should preserve 'a' second");
        } else {
            panic!("Expected Mul expression");
        }
    }

    /// Test that Matrix*Matrix multiplication preserves order (non-commutative)
    #[test]
    fn test_matrix_mul_preserves_order() {
        let mut ctx = Context::new();

        // Create two different matrices A and B
        let one = ctx.num(1);
        let two = ctx.num(2);
        let matrix_a = ctx.add(Expr::Matrix {
            rows: 1,
            cols: 1,
            data: vec![one],
        });
        let matrix_b = ctx.add(Expr::Matrix {
            rows: 1,
            cols: 1,
            data: vec![two],
        });

        // A*B should NOT be swapped even though ctx.add canonicalizes
        let mul_ab = ctx.add(Expr::Mul(matrix_a, matrix_b));
        if let Expr::Mul(l, r) = ctx.get(mul_ab) {
            assert_eq!(
                *l, matrix_a,
                "Matrix A*B: A should stay first (non-commutative)"
            );
            assert_eq!(
                *r, matrix_b,
                "Matrix A*B: B should stay second (non-commutative)"
            );
        } else {
            panic!("Expected Mul expression");
        }

        // B*A should also preserve its order
        let mul_ba = ctx.add(Expr::Mul(matrix_b, matrix_a));
        if let Expr::Mul(l, r) = ctx.get(mul_ba) {
            assert_eq!(
                *l, matrix_b,
                "Matrix B*A: B should stay first (non-commutative)"
            );
            assert_eq!(
                *r, matrix_a,
                "Matrix B*A: A should stay second (non-commutative)"
            );
        } else {
            panic!("Expected Mul expression");
        }
    }
}
