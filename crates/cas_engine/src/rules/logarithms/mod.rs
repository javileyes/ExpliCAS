//! Logarithm rules: evaluation, properties, inverse composition, and auto-expansion.
//!
//! This module is split into submodules:
//! - `properties`: Domain-aware expansion, contraction, abs/power rules, chain product
//! - `inverse`: Inverse composition rules (exp↔log), auto-expand log

mod inverse;
mod properties;

pub use inverse::{
    AutoExpandLogRule, ExponentialLogRule, LogExpInverseRule, LogInversePowerRule,
    LogPowerBaseRule, SplitLogExponentsRule,
};
pub use properties::{
    expand_logs, expand_logs_with_assumptions, LogAbsPowerRule, LogAbsSimplifyRule,
    LogChainProductRule, LogEvenPowerWithChainedAbsRule, LogExpansionRule,
};

use crate::define_rule;
use crate::ordering::compare_expr;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Zero};
use std::cmp::Ordering;

/// Helper: create log(base, arg) or ln(arg) depending on base.
/// If base is Constant::E, returns ln(arg) to preserve natural log notation.
/// If base is the sentinel for log10 (u32::MAX - 1), returns log(arg) with 1 arg.
pub(super) fn make_log(ctx: &mut Context, base: ExprId, arg: ExprId) -> ExprId {
    // Check for log10 sentinel first (before accessing ctx.get which would panic)
    let sentinel_log10 = ExprId::from_raw(u32::MAX - 1);
    if base == sentinel_log10 {
        return ctx.call_builtin(cas_ast::BuiltinFn::Log, vec![arg]);
    }
    if let Expr::Constant(cas_ast::Constant::E) = ctx.get(base) {
        ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![arg])
    } else {
        ctx.call_builtin(cas_ast::BuiltinFn::Log, vec![base, arg])
    }
}

/// Evaluate log_base(val) as a rational number using prime factorization.
///
/// Returns Some(ratio) if both base and val are powers of a common prime base:
/// - log(2, 8) = 3     because 2 = 2^1, 8 = 2^3 → ratio = 3/1
/// - log(8, 2) = 1/3   because 8 = 2^3, 2 = 2^1 → ratio = 1/3
/// - log(16, 8) = 3/4  because 16 = 2^4, 8 = 2^3 → ratio = 3/4
///
/// Returns None if the log cannot be expressed as a rational number.
fn eval_log_rational(
    base: &num_bigint::BigInt,
    val: &num_bigint::BigInt,
) -> Option<num_rational::BigRational> {
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::{One, Signed, Zero};

    // Guard: base > 1, val > 0
    let one = BigInt::one();
    if base <= &one || val.is_zero() || val.is_negative() || base.is_negative() {
        return None;
    }

    // Special case: val = 1 → log_b(1) = 0
    if val.is_one() {
        return Some(BigRational::zero());
    }

    // Special case: base == val → log_b(b) = 1
    if base == val {
        return Some(BigRational::one());
    }

    // Factorize both into prime -> exponent maps
    let fb = prime_exponent_map(base);
    let fv = prime_exponent_map(val);

    // Both must have the same set of primes
    if fb.keys().collect::<std::collections::HashSet<_>>()
        != fv.keys().collect::<std::collections::HashSet<_>>()
    {
        return None;
    }

    // The ratio exp_val / exp_base must be identical for all primes
    // log_base(val) = log(val) / log(base) = (sum of exp_val[p] * log(p)) / (sum of exp_base[p] * log(p))
    // This only works if exp_val[p] / exp_base[p] is constant for all p
    let mut ratio: Option<BigRational> = None;
    for (prime, exp_base) in &fb {
        let exp_val = fv.get(prime)?;
        if exp_base.is_zero() {
            return None; // Shouldn't happen, but guard
        }
        let r = BigRational::new(BigInt::from(*exp_val), BigInt::from(*exp_base));
        match &ratio {
            None => ratio = Some(r),
            Some(prev) if *prev == r => {}
            _ => return None, // Different ratios for different primes
        }
    }

    ratio
}

/// Compute prime factorization as a map of prime -> exponent
fn prime_exponent_map(n: &num_bigint::BigInt) -> HashMap<num_bigint::BigInt, u32> {
    use num_bigint::BigInt;
    use num_integer::Integer;
    use num_traits::{One, Signed, Zero};

    let mut result = HashMap::new();
    let mut n = n.abs();
    let one = BigInt::one();

    if n <= one {
        return result;
    }

    // Factor out 2s
    let mut count_2 = 0u32;
    while n.is_even() {
        count_2 += 1;
        n /= 2;
    }
    if count_2 > 0 {
        result.insert(BigInt::from(2), count_2);
    }

    // Factor out odd primes
    let mut d = BigInt::from(3);
    while &d * &d <= n {
        let mut count = 0u32;
        while (&n % &d).is_zero() {
            count += 1;
            n /= &d;
        }
        if count > 0 {
            result.insert(d.clone(), count);
        }
        d += 2;
    }

    // If n > 1 at this point, it's a prime
    if n > one {
        result.insert(n, 1);
    }

    result
}

use std::collections::HashMap;

define_rule!(EvaluateLogRule, "Evaluate Logarithms", |ctx, expr| {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let (fn_id, args) = (*fn_id, args.clone());
        use cas_ast::BuiltinFn;
        // Handle ln(x) as log(e, x)
        let (base, arg) = match ctx.builtin_of(fn_id) {
            Some(BuiltinFn::Ln) if args.len() == 1 => {
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                (e, args[0])
            }
            Some(BuiltinFn::Log) if args.len() == 2 => (args[0], args[1]),
            _ => return None,
        };

        // 1. log(b, 1) = 0, log(b, 0) = -infinity, log(b, neg) = undefined
        if let Expr::Number(n) = ctx.get(arg) {
            let n = n.clone();
            if n.is_one() {
                let zero = ctx.num(0);
                return Some(Rewrite::new(zero).desc("log(b, 1) = 0"));
            }
            if n.is_zero() {
                let inf = ctx.add(Expr::Constant(cas_ast::Constant::Infinity));
                let neg_inf = ctx.add(Expr::Neg(inf));
                return Some(Rewrite::new(neg_inf).desc("log(b, 0) = -infinity"));
            }
            if n < num_rational::BigRational::zero() {
                let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                return Some(Rewrite::new(undef).desc("log(b, neg) = undefined"));
            }

            // Check if n is a power of base (if base is a number)
            if let Expr::Number(b) = ctx.get(base) {
                let b = b.clone();
                // Try to evaluate log(base, val) as a rational number
                // This handles cases like:
                //   log(2, 8) = 3      (8 = 2^3)
                //   log(8, 2) = 1/3    (2 = 8^(1/3))
                //   log(16, 8) = 3/4   (16 = 2^4, 8 = 2^3, so log_16(8) = 3/4)
                if b.is_integer() && n.is_integer() {
                    let b_int = b.to_integer();
                    let n_int = n.to_integer();
                    if let Some(ratio) = eval_log_rational(&b_int, &n_int) {
                        let new_expr = ctx.add(Expr::Number(ratio.clone()));
                        return Some(
                            Rewrite::new(new_expr)
                                .desc_lazy(|| format!("log({}, {}) = {}", b, n, ratio)),
                        );
                    }
                }
            }
        }

        // 2. log(b, b) = 1
        if base == arg || ctx.get(base) == ctx.get(arg) {
            let one = ctx.num(1);
            return Some(Rewrite::new(one).desc("log(b, b) = 1"));
        }

        // 3. log(b, b^x) = x
        // NOTE: This inverse composition is now handled by LogExpInverseRule
        // which respects inv_trig policy (like arctan(tan(x)) → x).
        // Removed from here to avoid unconditional simplification.

        // 4. Expansion: log(b, x^y) = y * log(b, x)
        // Note: This overlaps with rule 3 if x == b. Rule 3 is more specific/simpler, so it should match first.
        // This rule is good for canonicalization.
        // GUARD: When x == b (inverse composition), only simplify if exponent is a number.
        // For variable exponents like log(e, e^x), let LogExpInverseRule handle with policy.
        if let Expr::Pow(p_base, p_exp) = ctx.get(arg) {
            let (p_base, p_exp) = (*p_base, *p_exp);
            let is_inverse_composition = p_base == base || ctx.get(p_base) == ctx.get(base);

            if is_inverse_composition {
                // log(b, b^n) where n is a number → n (always safe, like log(x, x^2) → 2)
                if matches!(ctx.get(p_exp), Expr::Number(_)) {
                    return Some(Rewrite::new(p_exp).desc("log(b, b^n) = n"));
                }
                // For variable exponents like log(e, e^x), skip and let LogExpInverseRule handle
                // with inv_trig policy check
            } else {
                // Non-inverse case: log(b, x^y) = y * log(b, x)
                //
                // MATHEMATICAL CORRECTNESS:
                // - For even integer exponents: ln(x^2) = ln(|x|^2) = 2·ln(|x|)
                //   This is valid for all x ≠ 0, no sign assumption needed.
                // - For odd/non-integer exponents: ln(x^y) = y·ln(x) requires x > 0
                //
                // Check if exponent is an even integer
                let is_even_integer = match ctx.get(p_exp) {
                    Expr::Number(n) if n.is_integer() => {
                        let int_val = n.to_integer();
                        let two: num_bigint::BigInt = 2.into();
                        &int_val % &two == 0.into() && int_val != 0.into()
                    }
                    _ => false,
                };

                if is_even_integer {
                    // Even exponent: ln(x^(2k)) = 2k·ln(|x|) - but this introduces abs()
                    //
                    // V2.14.45 ANTI-WORSEN GUARD:
                    // In Generic/Strict: BLOCK - this would introduce abs() without resolution
                    // Let LogEvenPowerWithChainedAbsRule handle this case in Assume mode
                    //
                    // NOTE: EvaluateLogRule uses define_rule! without parent_ctx, so we
                    // delegate to LogEvenPowerWithChainedAbsRule which has proper domain checks.
                    // Just skip this case here - the specialized rule will handle it.
                    return None;
                } else {
                    // Odd or non-integer exponent: requires x > 0
                    // Only allow in Generic/Assume modes, block in Strict
                    let log_inner = make_log(ctx, base, p_base);
                    let new_expr = smart_mul(ctx, p_exp, log_inner);
                    return Some(
                        Rewrite::new(new_expr)
                            .desc("log(b, x^y) = y * log(b, x)")
                            .assume(crate::assumptions::AssumptionEvent::positive_assumed(
                                ctx, p_base,
                            )),
                    );
                }
            }
        }

        // NOTE: Product/quotient expansions (log(xy) = log(x)+log(y), log(x/y) = log(x)-log(y))
        // are moved to LogExpansionRule which has domain_mode + value_domain gates.
        // These expansions require x > 0 and y > 0, and are NOT valid in complex domain with principal branch.
    }
    None
});

// =============================================================================
// LnEProductRule: ln(e*x) → 1 + ln(x)
// =============================================================================
// This is a SAFE, targeted expansion because ln(e) = 1 is a known constant.
// Unlike general LogExpansionRule, this doesn't risk explosion because it only
// extracts the known constant `e` factor.
//
// This enables residuals like `2*ln(e*u) - 2 - 2*ln(u)` to simplify:
// → 2*(1 + ln(u)) - 2 - 2*ln(u) → 2 + 2*ln(u) - 2 - 2*ln(u) → 0
// =============================================================================
define_rule!(LnEProductRule, "Factor e from ln Product", |ctx, expr| {
    // Match ln(arg)
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let (fn_id, args) = (*fn_id, args.clone());
        if ctx.builtin_of(fn_id) != Some(cas_ast::BuiltinFn::Ln) || args.len() != 1 {
            return None;
        }
        let arg = args[0];

        // Match Mul(e, x) or Mul(x, e) in the argument
        if let Expr::Mul(l, r) = ctx.get(arg) {
            let (l, r) = (*l, *r);
            let l_is_e = matches!(ctx.get(l), Expr::Constant(cas_ast::Constant::E));
            let r_is_e = matches!(ctx.get(r), Expr::Constant(cas_ast::Constant::E));

            // ln(e*x) → 1 + ln(x) or ln(x*e) → 1 + ln(x)
            let other = if l_is_e {
                Some(r)
            } else if r_is_e {
                Some(l)
            } else {
                None
            };

            if let Some(x) = other {
                let one = ctx.num(1);
                let ln_x = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![x]);
                let result = ctx.add(Expr::Add(one, ln_x));
                return Some(Rewrite::new(result).desc("ln(e*x) = 1 + ln(x)"));
            }
        }
    }
    None
});

// =============================================================================
// LnEDivRule: ln(x/e) → ln(x) - 1, ln(e/x) → 1 - ln(x)
// =============================================================================
// Companion to LnEProductRule. These are SAFE expansions because ln(e) = 1.
// This enables residuals like `2*ln(u/e) - 2*(ln(u)-1)` to simplify to 0.
// =============================================================================
define_rule!(LnEDivRule, "Factor e from ln Quotient", |ctx, expr| {
    // Match ln(arg)
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let (fn_id, args) = (*fn_id, args.clone());
        if ctx.builtin_of(fn_id) != Some(cas_ast::BuiltinFn::Ln) || args.len() != 1 {
            return None;
        }
        let arg = args[0];

        // Match Div(x, e) or Div(e, x) in the argument
        if let Expr::Div(num, den) = ctx.get(arg) {
            let (num, den) = (*num, *den);
            let num_is_e = matches!(ctx.get(num), Expr::Constant(cas_ast::Constant::E));
            let den_is_e = matches!(ctx.get(den), Expr::Constant(cas_ast::Constant::E));

            if den_is_e && !num_is_e {
                // ln(x/e) → ln(x) - 1
                let ln_x = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![num]);
                let one = ctx.num(1);
                let result = ctx.add(Expr::Sub(ln_x, one));
                return Some(Rewrite::new(result).desc("ln(x/e) = ln(x) - 1"));
            } else if num_is_e && !den_is_e {
                // ln(e/x) → 1 - ln(x)
                let one = ctx.num(1);
                let ln_x = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![den]);
                let result = ctx.add(Expr::Sub(one, ln_x));
                return Some(Rewrite::new(result).desc("ln(e/x) = 1 - ln(x)"));
            }
        }
    }
    None
});

/// Check if two expressions match (for telescoping condition)
/// Handles ln sentinel and structural equality
pub(super) fn exprs_match(ctx: &Context, e1: ExprId, e2: ExprId) -> bool {
    let sentinel = ExprId::from_raw(u32::MAX);

    // ln sentinel matches Constant::E
    if e1 == sentinel {
        return matches!(ctx.get(e2), Expr::Constant(cas_ast::Constant::E));
    }
    if e2 == sentinel {
        return matches!(ctx.get(e1), Expr::Constant(cas_ast::Constant::E));
    }

    // Direct ID match
    if e1 == e2 {
        return true;
    }

    // Structural comparison for same expression
    compare_expr(ctx, e1, e2) == Ordering::Equal
}

/// LogContractionRule: Contracts sums/differences of logs into single logs.
/// - ln(a) + ln(b) → ln(a*b)
/// - ln(a) - ln(b) → ln(a/b)
/// - log(b, x) + log(b, y) → log(b, x*y)  (same base required)
/// - log(b, x) - log(b, y) → log(b, x/y)
///
/// This rule REDUCES node count and is a valid simplification.
/// Unlike LogExpansionRule, this is registered by default.
pub struct LogContractionRule;

impl crate::rule::Rule for LogContractionRule {
    fn name(&self) -> &str {
        "Log Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::semantics::NormalFormGoal;

        // GATE: Don't contract logs when goal is ExpandedLog
        // This prevents undoing the effect of expand_log command
        if parent_ctx.goal() == NormalFormGoal::ExpandedLog {
            return None;
        }

        // GATE: Don't contract logs when in auto-expand mode
        // This prevents cycle with AutoExpandLogRule (expand→contract→expand→...)
        if parent_ctx.is_auto_expand() || parent_ctx.in_auto_expand_context() {
            return None;
        }

        let (lhs_opt, rhs_opt) = match ctx.get(expr) {
            Expr::Add(l, r) => (Some((*l, *r, true)), None),
            Expr::Sub(l, r) => (None, Some((*l, *r))),
            _ => (None, None),
        };

        // Case 1: ln(a) + ln(b) → ln(a*b) or log(b,x) + log(b,y) → log(b, x*y)
        if let Some((lhs, rhs, _is_add)) = lhs_opt {
            if let (Some((base_l, arg_l)), Some((base_r, arg_r))) =
                (extract_log_parts(ctx, lhs), extract_log_parts(ctx, rhs))
            {
                // Check bases are equal
                if bases_equal(ctx, base_l, base_r) {
                    let product = ctx.add(Expr::Mul(arg_l, arg_r));
                    // If base is sentinel (ln case), create ln(), otherwise log()
                    let sentinel = cas_ast::ExprId::from_raw(u32::MAX);
                    let new_expr = if base_l == sentinel {
                        ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![product])
                    } else {
                        make_log(ctx, base_l, product)
                    };
                    return Some(
                        crate::rule::Rewrite::new(new_expr).desc("ln(a) + ln(b) = ln(a*b)"),
                    );
                }
            }
        }

        // Case 2: ln(a) - ln(b) → ln(a/b) or log(b,x) - log(b,y) → log(b, x/y)
        if let Some((lhs, rhs)) = rhs_opt {
            if let (Some((base_l, arg_l)), Some((base_r, arg_r))) =
                (extract_log_parts(ctx, lhs), extract_log_parts(ctx, rhs))
            {
                // Check bases are equal
                if bases_equal(ctx, base_l, base_r) {
                    let quotient = ctx.add(Expr::Div(arg_l, arg_r));
                    // If base is sentinel (ln case), create ln(), otherwise log()
                    let sentinel = cas_ast::ExprId::from_raw(u32::MAX);
                    let new_expr = if base_l == sentinel {
                        ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![quotient])
                    } else {
                        make_log(ctx, base_l, quotient)
                    };
                    return Some(
                        crate::rule::Rewrite::new(new_expr).desc("ln(a) - ln(b) = ln(a/b)"),
                    );
                }
            }
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }
}

/// Extract (base, argument) from a log expression.
/// Returns (E, arg) for ln(arg), (base, arg) for log(base, arg), None otherwise.
pub(super) fn extract_log_parts(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        use cas_ast::BuiltinFn;
        match ctx.builtin_of(*fn_id) {
            Some(BuiltinFn::Ln) if args.len() == 1 => {
                return Some((cas_ast::ExprId::from_raw(u32::MAX), args[0]));
            }
            Some(BuiltinFn::Log) if args.len() == 2 => {
                return Some((args[0], args[1]));
            }
            _ => {}
        }
    }
    None
}

/// Check if two log bases are equal.
/// Handles ln (sentinel u32::MAX) specially.
pub(super) fn bases_equal(
    ctx: &cas_ast::Context,
    base_l: cas_ast::ExprId,
    base_r: cas_ast::ExprId,
) -> bool {
    let sentinel = cas_ast::ExprId::from_raw(u32::MAX);

    // Both are ln (sentinel)
    if base_l == sentinel && base_r == sentinel {
        return true;
    }

    // One is ln, other is explicit log(e, ...)
    if base_l == sentinel {
        if let Expr::Constant(cas_ast::Constant::E) = ctx.get(base_r) {
            return true;
        }
        return false;
    }
    if base_r == sentinel {
        if let Expr::Constant(cas_ast::Constant::E) = ctx.get(base_l) {
            return true;
        }
        return false;
    }

    // Both are explicit bases - check if equal by expression equality
    // For simplicity, only match if they're the same ExprId or both are Constant::E
    if base_l == base_r {
        return true;
    }

    // Check if both are e constant
    if let (Expr::Constant(cas_ast::Constant::E), Expr::Constant(cas_ast::Constant::E)) =
        (ctx.get(base_l), ctx.get(base_r))
    {
        return true;
    }

    false
}

// =============================================================================
// LogPerfectSquareRule: ln(f²) → 2·ln(|f|)
// =============================================================================
// Detects perfect-square arguments of log functions and factorizes them.
//
// Sub-detectors:
//   1. Polynomial trinomial: ln(4u²+12u+9) → 2·ln(|2u+3|) via discriminant=0
//   2. Trinomial via AST: ln(u⁴+2u²+1) → 2·ln(|u²+1|)
//   3. Even power: ln(u⁴) → 2·ln(|u²|)
//   4. Monomial: ln(4u²) → 2·ln(|2u|)
//   5. Div of squares: ln(u²/v²) → 2·ln(|u/v|)
//
// Domain: always wraps factor in |·| (safe); downstream |x|→x handles removal.
// =============================================================================
define_rule!(
    LogPerfectSquareRule,
    "Factor Perfect Square in Logarithm",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        use cas_ast::BuiltinFn;

        // Match ln(arg) or log(base, arg)
        let (fn_id, args) = match ctx.get(expr) {
            Expr::Function(fn_id, args) => (*fn_id, args.clone()),
            _ => return None,
        };

        let (log_base, arg) = match ctx.builtin_of(fn_id) {
            Some(BuiltinFn::Ln) if args.len() == 1 => {
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                (e, args[0])
            }
            Some(BuiltinFn::Log) if args.len() == 2 => (args[0], args[1]),
            _ => return None,
        };

        // Helper: build 2·log(base, |factor|) rewrite
        let build_result = |ctx: &mut Context,
                            factor: ExprId,
                            log_base: ExprId,
                            desc: &'static str|
         -> Option<Rewrite> {
            let abs_factor = ctx.call_builtin(BuiltinFn::Abs, vec![factor]);
            let log_inner = make_log(ctx, log_base, abs_factor);
            let two = ctx.num(2);
            let result = ctx.add(Expr::Mul(two, log_inner));
            Some(Rewrite::new(result).desc(desc))
        };

        // ----- Sub-detector 1: Polynomial trinomial (discriminant = 0) -----
        // Works for all coefficient-bearing trinomials like 4u²+12u+9
        if matches!(ctx.get(arg), Expr::Add(_, _) | Expr::Sub(_, _)) {
            // Polynomial-based approach (handles coefficients)
            let vars = cas_ast::collect_variables(ctx, arg);
            if vars.len() == 1 {
                let var = vars.iter().next()?;
                if let Ok(poly) = crate::polynomial::Polynomial::from_expr(ctx, arg, var) {
                    if poly.degree() == 2 && poly.coeffs.len() >= 3 {
                        let pa = poly.coeffs.get(2).cloned();
                        let pb = poly.coeffs.get(1).cloned();
                        let pc = poly.coeffs.first().cloned();

                        if let (Some(pa), Some(pb), Some(pc)) = (pa, pb, pc) {
                            use num_traits::Zero;
                            let four = num_rational::BigRational::from_integer(4.into());
                            let discriminant =
                                pb.clone() * pb.clone() - four * pa.clone() * pc.clone();

                            if discriminant.is_zero() {
                                if let Some(d) = crate::rules::algebra::roots::rational_sqrt(&pa) {
                                    let two_r = num_rational::BigRational::from_integer(2.into());
                                    let e = if d.is_zero() {
                                        crate::rules::algebra::roots::rational_sqrt(&pc)
                                            .unwrap_or_else(num_rational::BigRational::zero)
                                    } else {
                                        pb / (two_r * d.clone())
                                    };

                                    let var_expr = ctx.var(var);
                                    let d_expr = ctx.add(Expr::Number(d.clone()));
                                    let e_expr = ctx.add(Expr::Number(e.clone()));

                                    let one = num_rational::BigRational::from_integer(1.into());
                                    let dx = if d == one {
                                        var_expr
                                    } else {
                                        crate::rules::algebra::helpers::smart_mul(
                                            ctx, d_expr, var_expr,
                                        )
                                    };

                                    let linear = if e.is_zero() {
                                        dx
                                    } else {
                                        ctx.add(Expr::Add(dx, e_expr))
                                    };

                                    return build_result(
                                        ctx,
                                        linear,
                                        log_base,
                                        "ln((A+B)²) = 2·ln(|A+B|)",
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Fallback: AST-level trinomial matching for multivariate cases
            if let Some((a, b, is_sub)) =
                crate::rules::polynomial::try_match_perfect_square_trinomial(ctx, arg)
            {
                let factor = if is_sub {
                    ctx.add(Expr::Sub(a, b))
                } else {
                    ctx.add(Expr::Add(a, b))
                };
                let desc = if is_sub {
                    "ln((A−B)²) = 2·ln(|A−B|)"
                } else {
                    "ln((A+B)²) = 2·ln(|A+B|)"
                };
                return build_result(ctx, factor, log_base, desc);
            }
        }

        // ----- Sub-detector 2: Even power Pow(base, 2k) -----
        // ln(u⁴) → 2·ln(|u²|), ln(sin⁴(u)) → 2·ln(|sin²(u)|)
        if let Expr::Pow(p_base, p_exp) = ctx.get(arg) {
            let (p_base, p_exp) = (*p_base, *p_exp);
            // Skip inverse composition: ln(e^x) handled by other rules
            if p_base == log_base || ctx.get(p_base) == ctx.get(log_base) {
                return None;
            }
            if let Expr::Number(n) = ctx.get(p_exp) {
                let n = n.clone();
                if n.is_integer() {
                    let int_val = n.to_integer();
                    let two_bi: num_bigint::BigInt = 2.into();
                    if &int_val % &two_bi == 0.into() && int_val > 0.into() {
                        let half = &int_val / &two_bi;
                        let half_rat = num_rational::BigRational::from_integer(half.clone());
                        let factor = if half == 1.into() {
                            p_base
                        } else {
                            let half_id = ctx.add(Expr::Number(half_rat));
                            ctx.add(Expr::Pow(p_base, half_id))
                        };
                        return build_result(ctx, factor, log_base, "ln(x^(2k)) = 2·ln(|x^k|)");
                    }
                }
            }
        }

        // ----- Sub-detector 3: Monomial Mul(n, Pow(base, 2k)) where n is perfect square -----
        // ln(4u²) → 2·ln(|2u|)
        if let Expr::Mul(l, r) = ctx.get(arg) {
            let (l, r) = (*l, *r);
            for (maybe_coeff, maybe_pow) in [(l, r), (r, l)] {
                if let Expr::Number(coeff) = ctx.get(maybe_coeff) {
                    if coeff.is_integer() && num_traits::Signed::is_positive(coeff) {
                        let coeff_int = coeff.to_integer();
                        let coeff_root = coeff_int.sqrt();
                        if &coeff_root * &coeff_root == coeff_int {
                            if let Some(pow_root) =
                                crate::rules::polynomial::extract_square_root_of_term(
                                    ctx, maybe_pow,
                                )
                            {
                                let root_num = ctx.add(Expr::Number(
                                    num_rational::BigRational::from_integer(coeff_root),
                                ));
                                let factor = crate::rules::algebra::helpers::smart_mul(
                                    ctx, root_num, pow_root,
                                );
                                return build_result(
                                    ctx,
                                    factor,
                                    log_base,
                                    "ln((c·x)²) = 2·ln(|c·x|)",
                                );
                            }
                        }
                    }
                }
            }
        }

        // ----- Sub-detector 4: Div of squares: ln(a²/b²) → 2·ln(|a/b|) -----
        if let Expr::Div(num, den) = ctx.get(arg) {
            let (num, den) = (*num, *den);
            if let (Some(num_root), Some(den_root)) = (
                crate::rules::polynomial::extract_square_root_of_term(ctx, num),
                crate::rules::polynomial::extract_square_root_of_term(ctx, den),
            ) {
                let factor = ctx.add(Expr::Div(num_root, den_root));
                return build_result(ctx, factor, log_base, "ln(a²/b²) = 2·ln(|a/b|)");
            }
        }

        None
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    // V2.14.45: LogPowerBaseRule MUST come BEFORE LogEvenPowerRule
    // Otherwise log(x^2, x^6) gets expanded to 6·log(x², |x|) before simplifying to 3
    simplifier.add_rule(Box::new(LogPowerBaseRule)); // log(a^m, a^n) → n/m

    // LogAbsPowerRule: ln(|u|^n) → n·ln(|u|) - highest priority (15)
    // MUST come BEFORE AbsSquareRule which would turn |u|^2 → u^2 and lose the abs
    simplifier.add_rule(Box::new(LogAbsPowerRule));

    // V2.14.20: LogEvenPowerWithChainedAbsRule handles ln(x^even) with ChainedRewrite
    // Has higher priority (10) than EvaluateLogRule (0) so matches first
    simplifier.add_rule(Box::new(LogEvenPowerWithChainedAbsRule));

    // LogPerfectSquareRule: ln(A² ± 2AB + B²) → 2·ln(|A ± B|)
    // Factor perfect-square trinomials inside log arguments (e.g. u⁴+2u²+1 → (u²+1)²)
    simplifier.add_rule(Box::new(LogPerfectSquareRule));

    simplifier.add_rule(Box::new(EvaluateLogRule));

    // LnEProductRule: ln(e*x) → 1 + ln(x) - safe targeted expansion
    // This enables residuals like `2*ln(e*u) - 2 - 2*ln(u)` to simplify to 0
    simplifier.add_rule(Box::new(LnEProductRule));
    // LnEDivRule: ln(x/e) → ln(x) - 1, ln(e/x) → 1 - ln(x)
    // Companion to LnEProductRule for quotient cases
    simplifier.add_rule(Box::new(LnEDivRule));

    // NOTE: LogExpansionRule removed from auto-registration.
    // Log expansion increases node count (ln(xy) → ln(x) + ln(y)) and is not always desirable.
    // Use the `expand_log` command for explicit expansion.
    // simplifier.add_rule(Box::new(LogExpansionRule));

    // LogAbsSimplifyRule: ln(|x|) → ln(x) when x > 0
    // Must be BEFORE LogContractionRule to catch `ln(|x|) - ln(x)` before it becomes `ln(|x|/x)`
    simplifier.add_rule(Box::new(LogAbsSimplifyRule));

    // LogChainProductRule: log(b,a)*log(a,c) → log(b,c) (telescoping)
    // Must be BEFORE LogContractionRule
    simplifier.add_rule(Box::new(LogChainProductRule));

    // LogContractionRule DOES reduce node count (ln(a)+ln(b) → ln(ab)) - valid simplification
    simplifier.add_rule(Box::new(LogContractionRule));

    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(SplitLogExponentsRule));
    simplifier.add_rule(Box::new(LogInversePowerRule));
    simplifier.add_rule(Box::new(LogExpInverseRule));

    // AutoExpandLogRule: auto-expand log products/quotients when log_expand_policy=Auto
    simplifier.add_rule(Box::new(AutoExpandLogRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn test_log_one() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(x, 1) -> 0
        let expr = parse("log(x, 1)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn test_log_base_base() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(x, x) -> 1
        let expr = parse("log(x, x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "1"
        );
    }

    #[test]
    fn test_log_inverse() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(x, x^2) -> 2
        let expr = parse("log(x, x^2)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "2"
        );
    }

    #[test]
    fn test_log_expansion() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(b, x^y) -> y * log(b, x)
        let expr = parse("log(2, x^3)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "3 * log(2, x)"
        );
    }

    #[test]
    fn test_log_product() {
        let mut ctx = Context::new();
        let rule = LogExpansionRule;
        // log(b, x*y) -> log(b, x) + log(b, y) (requires Assume mode for variables)
        let expr = parse("log(2, x * y)", &mut ctx).unwrap();
        // Create parent context with Assume mode (allows expansion with warning)
        let parent_ctx = crate::parent_context::ParentContext::root()
            .with_domain_mode(crate::domain::DomainMode::Assume);
        let rewrite = rule.apply(&mut ctx, expr, &parent_ctx).unwrap();
        let res = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(res.contains("log(2, x)"));
        assert!(res.contains("log(2, y)"));
        assert!(res.contains("+"));
    }

    #[test]
    fn test_log_quotient() {
        let mut ctx = Context::new();
        let rule = LogExpansionRule;
        // log(b, x/y) -> log(b, x) - log(b, y) (requires Assume mode for variables)
        let expr = parse("log(2, x / y)", &mut ctx).unwrap();
        // Create parent context with Assume mode (allows expansion with warning)
        let parent_ctx = crate::parent_context::ParentContext::root()
            .with_domain_mode(crate::domain::DomainMode::Assume);
        let rewrite = rule.apply(&mut ctx, expr, &parent_ctx).unwrap();
        let res = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(res.contains("log(2, x)"));
        assert!(res.contains("log(2, y)"));
        assert!(res.contains("-"));
    }

    #[test]
    fn test_ln_e() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // ln(e) -> 1
        // Note: ln(e) parses to log(e, e)
        let expr = parse("ln(e)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "1"
        );
    }
}
