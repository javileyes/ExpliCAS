// ========== Common Expression Predicates ==========
// These functions were previously duplicated across multiple files.
// Now consolidated here for consistency and maintainability.

use super::pi::extract_rational_pi_multiple;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::{One, Signed};

/// Check if expression is the number 1
pub(crate) fn is_one(ctx: &Context, expr: ExprId) -> bool {
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
    // Use depth-limited version with max 50 levels to prevent stack overflow
    prove_nonzero_depth(ctx, expr, 50)
}

/// Internal prove_nonzero with explicit depth limit.
fn prove_nonzero_depth(ctx: &Context, expr: ExprId, depth: usize) -> crate::domain::Proof {
    use crate::domain::Proof;
    use num_traits::Zero;

    // Depth guard: return Unknown if we've recursed too deep
    if depth == 0 {
        return Proof::Unknown;
    }

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
        // Hold: transparent — Hold(x) ≠ 0 iff x ≠ 0
        Expr::Neg(a) | Expr::Hold(a) => prove_nonzero_depth(ctx, *a, depth - 1),

        // Mul: a*b ≠ 0 iff a ≠ 0 AND b ≠ 0
        Expr::Mul(a, b) => {
            let proof_a = prove_nonzero_depth(ctx, *a, depth - 1);
            let proof_b = prove_nonzero_depth(ctx, *b, depth - 1);

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
                    return prove_nonzero_depth(ctx, *base, depth - 1);
                }
            }
            Proof::Unknown
        }

        // Div: a/b ≠ 0 iff a ≠ 0 (assuming b ≠ 0 for the expression to be defined)
        Expr::Div(a, b) => {
            let proof_a = prove_nonzero_depth(ctx, *a, depth - 1);
            let proof_b = prove_nonzero_depth(ctx, *b, depth - 1);

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
        Expr::Function(fn_id, args)
            if (ctx.is_builtin(*fn_id, BuiltinFn::Ln)
                || ctx.is_builtin(*fn_id, BuiltinFn::Log))
                && args.len() == 1 =>
        {
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

        // sin(k·π): zero iff k is integer, non-zero iff k is rational non-integer
        // This enables cancellation of sin(π/9)/sin(π/9) without requiring assumptions
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sin) && args.len() == 1 =>
        {
            // Try to extract k from sin(k·π)
            if let Some(k) = extract_rational_pi_multiple(ctx, args[0]) {
                // k.is_integer() checks if denominator == 1 (works on reduced form)
                if k.is_integer() {
                    Proof::Disproven // sin(n·π) = 0 for any integer n
                } else {
                    Proof::Proven // sin(k·π) ≠ 0 for non-integer rational k
                }
            } else {
                // Not a rational multiple of π (e.g., sin(a) for symbolic a)
                Proof::Unknown
            }
        }

        // Variables, other functions: UNKNOWN (conservative)
        _ => Proof::Unknown,
    }
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
    // Use depth-limited version with max 50 levels to prevent stack overflow
    prove_positive_depth(ctx, expr, value_domain, 50)
}

/// Internal prove_positive with explicit depth limit.
fn prove_positive_depth(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
    depth: usize,
) -> crate::domain::Proof {
    use crate::domain::Proof;
    use crate::semantics::ValueDomain;
    use num_traits::Zero;

    // Depth guard: return Unknown if we've recursed too deep
    if depth == 0 {
        return Proof::Unknown;
    }

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

            let proof_a = prove_positive_depth(ctx, *a, value_domain, depth - 1);
            let proof_b = prove_positive_depth(ctx, *b, value_domain, depth - 1);

            match (proof_a, proof_b) {
                (Proof::Proven, Proof::Proven) => Proof::Proven,
                // If either is ≤ 0, we can't easily conclude (could be positive if both negative)
                (Proof::Disproven, _) | (_, Proof::Disproven) => Proof::Unknown,
                _ => Proof::Unknown,
            }
        }

        // Div: a/b > 0 if (a>0 AND b>0) OR (a<0 AND b<0)
        Expr::Div(a, b) => {
            let proof_a = prove_positive_depth(ctx, *a, value_domain, depth - 1);
            let proof_b = prove_positive_depth(ctx, *b, value_domain, depth - 1);

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
            let base_positive = prove_positive_depth(ctx, *base, value_domain, depth - 1);

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
                        let base_nonzero = prove_nonzero_depth(ctx, *base, depth - 1);
                        if base_nonzero == Proof::Proven {
                            return Proof::Proven;
                        }
                    }
                }
            }
            Proof::Unknown
        }

        // abs(x): always ≥ 0, but only > 0 if x ≠ 0
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 =>
        {
            let inner_nonzero = prove_nonzero_depth(ctx, args[0], depth - 1);
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
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Exp) && args.len() == 1 =>
        {
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
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            prove_positive_depth(ctx, args[0], value_domain, depth - 1)
        }

        // Neg: -x > 0 iff x < 0
        // If x is proven positive (> 0), then -x is proven negative (< 0), so Disproven
        // If x is proven negative (< 0), then -x is proven positive (> 0), so Proven
        Expr::Neg(inner) => {
            let inner_proof = prove_positive_depth(ctx, *inner, value_domain, depth - 1);
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

        // Hold: transparent — Hold(x) > 0 iff x > 0
        Expr::Hold(inner) => prove_positive_depth(ctx, *inner, value_domain, depth - 1),

        // Variables, other functions: UNKNOWN (conservative)
        _ => Proof::Unknown,
    }
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
pub(crate) fn prove_nonnegative(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> crate::domain::Proof {
    // Use depth-limited version with max 50 levels to prevent stack overflow
    prove_nonnegative_depth(ctx, expr, value_domain, 50)
}

/// Internal prove_nonnegative with explicit depth limit.
fn prove_nonnegative_depth(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
    depth: usize,
) -> crate::domain::Proof {
    use crate::domain::Proof;
    use crate::semantics::ValueDomain;
    use num_traits::Zero;

    // Depth guard: return Unknown if we've recursed too deep
    if depth == 0 {
        return Proof::Unknown;
    }

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
                let base_positive = prove_positive_depth(ctx, *base, value_domain, depth - 1);
                if base_positive == Proof::Proven {
                    return Proof::Proven;
                }
            }

            Proof::Unknown
        }

        // abs(x): always ≥ 0
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 =>
        {
            Proof::Proven
        }

        // sqrt(x): if defined, result is ≥ 0 (by principal root convention)
        // But we can't prove sqrt is defined without proving arg ≥ 0 (circular)
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            // If the arg is provably non-negative, sqrt(arg) ≥ 0
            prove_nonnegative_depth(ctx, args[0], value_domain, depth - 1)
        }

        // exp(x) > 0 for all real x, hence ≥ 0
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Exp) && args.len() == 1 =>
        {
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
            let proof_a = prove_nonnegative_depth(ctx, *a, value_domain, depth - 1);
            let proof_b = prove_nonnegative_depth(ctx, *b, value_domain, depth - 1);

            match (proof_a, proof_b) {
                (Proof::Proven, Proof::Proven) => Proof::Proven,
                _ => Proof::Unknown, // Can't easily prove otherwise
            }
        }

        // Neg: -x ≥ 0 iff x ≤ 0 - check if x is ≤ 0
        Expr::Neg(inner) => {
            // If inner is provably ≤ 0 (disproven as non-negative), then -inner ≥ 0
            let inner_proof = prove_nonnegative_depth(ctx, *inner, value_domain, depth - 1);
            match inner_proof {
                Proof::Disproven => Proof::Proven, // -(-3) = 3 ≥ 0
                _ => Proof::Unknown,
            }
        }

        // Hold: transparent — Hold(x) ≥ 0 iff x ≥ 0
        Expr::Hold(inner) => prove_nonnegative_depth(ctx, *inner, value_domain, depth - 1),

        // Variables, other functions: UNKNOWN (conservative)
        _ => Proof::Unknown,
    }
}
