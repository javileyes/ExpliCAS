use crate::expr_nary::build_balanced_mul;
use crate::poly_gcd_mode::GcdMode;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberTheoryDispatch {
    Simple(NumberTheorySimpleRewrite),
    PolyGcd {
        lhs: ExprId,
        rhs: ExprId,
        mode: GcdMode,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsecutiveFactorialRatioRewrite {
    pub rewritten: ExprId,
    pub factorial_arg_requires_nonnegative: ExprId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PascalChooseRewrite {
    pub rewritten: ExprId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChooseSymmetryRewrite {
    pub rewritten: ExprId,
}

/// Simple number-theory rewrite produced from a named function call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberTheorySimpleRewrite {
    Unary {
        name: &'static str,
        arg: ExprId,
        result: ExprId,
    },
    Binary {
        name: &'static str,
        lhs: ExprId,
        rhs: ExprId,
        result: ExprId,
    },
}

impl NumberTheorySimpleRewrite {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Unary { name, .. } | Self::Binary { name, .. } => name,
        }
    }

    pub fn result(&self) -> ExprId {
        match self {
            Self::Unary { result, .. } | Self::Binary { result, .. } => *result,
        }
    }
}

/// Evaluate simple named number-theory calls.
///
/// This handles integer-only/safe operations:
/// - `gcd` (integer case only; polynomial fallback lives in engine)
/// - `lcm`, `mod`
/// - `prime_factors` / `factors`
/// - `fact` / `factorial`
/// - `choose` / `nCr`
/// - `perm` / `nPr`
pub fn try_eval_simple_number_theory_call(
    ctx: &mut Context,
    name: &str,
    args: &[ExprId],
) -> Option<NumberTheorySimpleRewrite> {
    match name {
        "gcd" if args.len() >= 2 => {
            // Fold over ALL arguments: gcd(a, b, c, …) = gcd(gcd(a, b), c, …). Using only the
            // first two silently returned wrong answers (gcd(8, 12, 6) → 4 instead of 2).
            let mut result = args[0];
            for &arg in &args[1..] {
                result = compute_integer_gcd_expr(ctx, result, arg)?;
            }
            Some(NumberTheorySimpleRewrite::Binary {
                name: "gcd",
                lhs: args[0],
                rhs: args[1],
                result,
            })
        }
        "lcm" if args.len() >= 2 => {
            let mut result = args[0];
            for &arg in &args[1..] {
                result = compute_lcm_expr(ctx, result, arg)?;
            }
            Some(NumberTheorySimpleRewrite::Binary {
                name: "lcm",
                lhs: args[0],
                rhs: args[1],
                result,
            })
        }
        "mod" if args.len() == 2 => {
            let result = compute_mod_expr(ctx, args[0], args[1])?;
            Some(NumberTheorySimpleRewrite::Binary {
                name: "mod",
                lhs: args[0],
                rhs: args[1],
                result,
            })
        }
        "prime_factors" | "factors" if args.len() == 1 => {
            let result = compute_prime_factors_expr(ctx, args[0])?;
            Some(NumberTheorySimpleRewrite::Unary {
                name: "factors",
                arg: args[0],
                result,
            })
        }
        "isprime" | "is_prime" if args.len() == 1 => {
            let result = compute_isprime_expr(ctx, args[0])?;
            Some(NumberTheorySimpleRewrite::Unary {
                name: "isprime",
                arg: args[0],
                result,
            })
        }
        "nextprime" if args.len() == 1 => {
            let result = compute_nextprime_expr(ctx, args[0])?;
            Some(NumberTheorySimpleRewrite::Unary {
                name: "nextprime",
                arg: args[0],
                result,
            })
        }
        "prevprime" if args.len() == 1 => {
            let result = compute_prevprime_expr(ctx, args[0])?;
            Some(NumberTheorySimpleRewrite::Unary {
                name: "prevprime",
                arg: args[0],
                result,
            })
        }
        "totient" | "phi" | "eulerphi" if args.len() == 1 => {
            let result = compute_totient_expr(ctx, args[0])?;
            Some(NumberTheorySimpleRewrite::Unary {
                name: "totient",
                arg: args[0],
                result,
            })
        }
        "fact" | "factorial" if args.len() == 1 => {
            let result = compute_factorial_expr(ctx, args[0])?;
            Some(NumberTheorySimpleRewrite::Unary {
                name: "fact",
                arg: args[0],
                result,
            })
        }
        "choose" | "nCr" if args.len() == 2 => {
            let result = compute_choose_expr(ctx, args[0], args[1])?;
            Some(NumberTheorySimpleRewrite::Binary {
                name: "choose",
                lhs: args[0],
                rhs: args[1],
                result,
            })
        }
        "perm" | "nPr" if args.len() == 2 => {
            let result = compute_perm_expr(ctx, args[0], args[1])?;
            Some(NumberTheorySimpleRewrite::Binary {
                name: "perm",
                lhs: args[0],
                rhs: args[1],
                result,
            })
        }
        _ => None,
    }
}

/// Dispatch a number-theory function call into either:
/// - a fully solved simple rewrite, or
/// - a polynomial-gcd fallback request (`gcd` symbolic/poly case).
pub fn dispatch_number_theory_call(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<NumberTheoryDispatch> {
    let (name, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (ctx.sym_name(*fn_id).to_string(), args.clone()),
        _ => return None,
    };

    if let Some(simple) = try_eval_simple_number_theory_call(ctx, name.as_str(), &args) {
        return Some(NumberTheoryDispatch::Simple(simple));
    }

    if name == "gcd" && args.len() >= 2 {
        use crate::poly_gcd_mode::parse_gcd_mode;
        let explicit_mode = if args.len() >= 3 {
            Some(parse_gcd_mode(ctx, args[2]))
        } else {
            None
        };
        let mode = select_poly_gcd_mode(ctx, args[0], args[1], explicit_mode);
        return Some(NumberTheoryDispatch::PolyGcd {
            lhs: args[0],
            rhs: args[1],
            mode,
        });
    }

    None
}

pub fn try_rewrite_pascal_choose_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PascalChooseRewrite> {
    let (left, right) = match ctx.get(expr) {
        Expr::Add(left, right) => (*left, *right),
        _ => return None,
    };

    let (n_left, k_left) = extract_integer_choose_call(ctx, left)?;
    let (n_right, k_right) = extract_integer_choose_call(ctx, right)?;
    if n_left != n_right {
        return None;
    }

    let (lower_k, upper_k) = if k_left <= k_right {
        (k_left, k_right)
    } else {
        (k_right, k_left)
    };
    if upper_k - &lower_k != BigInt::one() {
        return None;
    }

    let rewritten_n = n_left + BigInt::one();
    let rewritten_k = lower_k + BigInt::one();
    let n_expr = ctx.add(Expr::Number(BigRational::from_integer(rewritten_n)));
    let k_expr = ctx.add(Expr::Number(BigRational::from_integer(rewritten_k)));
    let rewritten = ctx.call("choose", vec![n_expr, k_expr]);

    Some(PascalChooseRewrite { rewritten })
}

pub fn try_rewrite_choose_symmetry_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ChooseSymmetryRewrite> {
    let (n, k) = extract_integer_choose_call(ctx, expr)?;
    if n.is_negative() || k.is_negative() || k > n {
        return None;
    }

    let complement = &n - &k;
    if k >= complement {
        return None;
    }

    let n_expr = ctx.add(Expr::Number(BigRational::from_integer(n)));
    let k_expr = ctx.add(Expr::Number(BigRational::from_integer(complement)));
    let rewritten = ctx.call("choose", vec![n_expr, k_expr]);

    Some(ChooseSymmetryRewrite { rewritten })
}

fn extract_integer_choose_call(ctx: &Context, expr: ExprId) -> Option<(BigInt, BigInt)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 2 && matches!(ctx.sym_name(*fn_id), "choose" | "nCr") =>
        {
            Some((
                extract_integer_bigint(ctx, args[0])?,
                extract_integer_bigint(ctx, args[1])?,
            ))
        }
        _ => None,
    }
}

fn extract_factorial_call_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.sym_name(*fn_id), "fact" | "factorial") =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn extract_additive_base_and_integer_offset(ctx: &Context, expr: ExprId) -> Option<(ExprId, i64)> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            if let Some(offset) = get_integer_exponent(ctx, *left) {
                return Some((*right, offset));
            }
            if let Some(offset) = get_integer_exponent(ctx, *right) {
                return Some((*left, offset));
            }
            None
        }
        Expr::Sub(left, right) => get_integer_exponent(ctx, *right).map(|offset| (*left, -offset)),
        _ => Some((expr, 0)),
    }
}

fn rebuild_base_with_integer_offset(ctx: &mut Context, base: ExprId, offset: i64) -> ExprId {
    if offset == 0 {
        return base;
    }

    let amount = ctx.num(offset.checked_abs().expect("factorial offset fits in i64"));
    if offset > 0 {
        ctx.add(Expr::Add(base, amount))
    } else {
        ctx.add(Expr::Sub(base, amount))
    }
}

pub fn try_rewrite_consecutive_factorial_ratio_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ConsecutiveFactorialRatioRewrite> {
    let (num, den) = crate::expr_destructure::as_div(ctx, expr)?;
    let num_arg = extract_factorial_call_arg(ctx, num)?;
    let den_arg = extract_factorial_call_arg(ctx, den)?;

    let (num_base, num_offset) = extract_additive_base_and_integer_offset(ctx, num_arg)?;
    let (den_base, den_offset) = extract_additive_base_and_integer_offset(ctx, den_arg)?;
    if compare_expr(ctx, num_base, den_base) != std::cmp::Ordering::Equal {
        return None;
    }

    let gap = num_offset - den_offset;
    if gap <= 0 {
        return None;
    }

    let rewritten = if gap == 1 {
        num_arg
    } else {
        let mut factors = Vec::with_capacity(gap as usize);
        for shift in 1..=gap {
            let factor_offset = den_offset + shift;
            let factor = if factor_offset == num_offset {
                num_arg
            } else {
                rebuild_base_with_integer_offset(ctx, den_base, factor_offset)
            };
            factors.push(factor);
        }
        build_balanced_mul(ctx, &factors)
    };

    Some(ConsecutiveFactorialRatioRewrite {
        rewritten,
        factorial_arg_requires_nonnegative: den_arg,
    })
}

/// Check if expression contains a `poly_result(...)` reference.
pub fn contains_poly_result(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            if ctx.is_builtin(*fn_id, BuiltinFn::PolyResult) {
                return true;
            }
            if ctx.is_builtin(*fn_id, BuiltinFn::Hold) && !args.is_empty() {
                return contains_poly_result(ctx, args[0]);
            }
            args.iter().any(|&arg| contains_poly_result(ctx, arg))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            contains_poly_result(ctx, *l) || contains_poly_result(ctx, *r)
        }
        Expr::Pow(base, exp) => contains_poly_result(ctx, *base) || contains_poly_result(ctx, *exp),
        Expr::Neg(inner) => contains_poly_result(ctx, *inner),
        _ => false,
    }
}

/// Extract integer exponent from expression.
pub fn get_integer_exponent(ctx: &Context, exp: ExprId) -> Option<i64> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().try_into().ok()
            } else {
                None
            }
        }
        Expr::Neg(inner) => get_integer_exponent(ctx, *inner).map(|k| -k),
        _ => None,
    }
}

/// Check if expression has large unexpanded powers (exponent > 2) over non-atomic bases.
pub fn has_large_unexpanded_power(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if let Some(n) = get_integer_exponent(ctx, *exp) {
                if n > 2 && !matches!(ctx.get(*base), Expr::Variable(_) | Expr::Number(_)) {
                    return true;
                }
            }
            has_large_unexpanded_power(ctx, *base) || has_large_unexpanded_power(ctx, *exp)
        }
        Expr::Function(fn_id, args) => {
            if ctx.is_builtin(*fn_id, BuiltinFn::PolyResult) {
                return false;
            }
            args.iter().any(|&arg| has_large_unexpanded_power(ctx, arg))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            has_large_unexpanded_power(ctx, *l) || has_large_unexpanded_power(ctx, *r)
        }
        Expr::Neg(inner) => has_large_unexpanded_power(ctx, *inner),
        _ => false,
    }
}

/// Check whether an expression is `sqrt(n)` for an exact integer `n`.
///
/// Accepted forms:
/// - `n^(1/2)`
/// - `sqrt(n)`
pub fn is_sqrt_of_integer_expr(ctx: &Context, id: ExprId, n: i64) -> bool {
    let target = BigRational::from_integer(BigInt::from(n));
    let half = BigRational::new(1.into(), 2.into());

    match ctx.get(id) {
        Expr::Pow(base, exp) => {
            matches!((ctx.get(*base), ctx.get(*exp)), (Expr::Number(b), Expr::Number(e)) if *b == target && *e == half)
        }
        Expr::Function(name, args) if ctx.is_builtin(*name, BuiltinFn::Sqrt) && args.len() == 1 => {
            matches!(ctx.get(args[0]), Expr::Number(v) if *v == target)
        }
        _ => false,
    }
}

/// Select effective polynomial GCD mode, honoring explicit mode when provided.
///
/// Auto-mode heuristic:
/// - use `Modp` when expression references `poly_result(...)` or contains
///   large unexpanded powers over non-atomic bases;
/// - otherwise use `Structural`.
pub fn select_poly_gcd_mode(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    explicit_mode: Option<GcdMode>,
) -> GcdMode {
    if let Some(mode) = explicit_mode {
        return mode;
    }

    let needs_modp = contains_poly_result(ctx, a)
        || contains_poly_result(ctx, b)
        || has_large_unexpanded_power(ctx, a)
        || has_large_unexpanded_power(ctx, b);

    if needs_modp {
        GcdMode::Modp
    } else {
        GcdMode::Structural
    }
}

/// Extract exact integer value from expression.
pub fn extract_integer_bigint(ctx: &Context, expr: ExprId) -> Option<BigInt> {
    crate::expr_extract::extract_integer_exact(ctx, expr)
}

/// Trial-division search/test stay fast below this bound; larger inputs decline
/// to an honest residual rather than block on a slow loop.
const NUMBER_THEORY_PRIME_SEARCH_CAP: i64 = 1_000_000_000_000;

/// Exact primality by trial division (deterministic, no f64). `n < 2 ⇒ false`.
fn is_prime_bigint(n: &BigInt) -> bool {
    let two = BigInt::from(2);
    if *n < two {
        return false;
    }
    if *n == two {
        return true;
    }
    if n.is_even() {
        return false;
    }
    let mut d = BigInt::from(3);
    while &d * &d <= *n {
        if (n % &d).is_zero() {
            return false;
        }
        d += 2;
    }
    true
}

fn within_prime_search_cap(n: &BigInt) -> bool {
    n.abs() <= BigInt::from(NUMBER_THEORY_PRIME_SEARCH_CAP)
}

fn integer_result(ctx: &mut Context, value: BigInt) -> ExprId {
    ctx.add(Expr::Number(BigRational::from_integer(value)))
}

/// `isprime(n)` → `1` if prime, `0` otherwise (the engine has no boolean type).
pub fn compute_isprime_expr(ctx: &mut Context, n: ExprId) -> Option<ExprId> {
    let val = extract_integer_bigint(ctx, n)?;
    if !within_prime_search_cap(&val) {
        return None;
    }
    Some(ctx.num(if is_prime_bigint(&val) { 1 } else { 0 }))
}

/// Smallest prime strictly greater than `n`.
pub fn compute_nextprime_expr(ctx: &mut Context, n: ExprId) -> Option<ExprId> {
    let val = extract_integer_bigint(ctx, n)?;
    if !within_prime_search_cap(&val) {
        return None;
    }
    let mut candidate = if val < BigInt::from(2) {
        BigInt::from(2)
    } else {
        &val + 1
    };
    while !is_prime_bigint(&candidate) {
        candidate += 1;
    }
    Some(integer_result(ctx, candidate))
}

/// Largest prime strictly less than `n` (declines when none exists, i.e. `n ≤ 2`).
pub fn compute_prevprime_expr(ctx: &mut Context, n: ExprId) -> Option<ExprId> {
    let val = extract_integer_bigint(ctx, n)?;
    if !within_prime_search_cap(&val) || val <= BigInt::from(2) {
        return None;
    }
    let mut candidate = &val - 1;
    let two = BigInt::from(2);
    while candidate >= two && !is_prime_bigint(&candidate) {
        candidate -= 1;
    }
    (candidate >= BigInt::from(2)).then(|| integer_result(ctx, candidate))
}

/// Euler's totient `φ(n) = n·∏(1 − 1/p)` over the distinct primes `p | n`, by exact
/// integer factorization. Declines for `n < 1`.
pub fn compute_totient_expr(ctx: &mut Context, n: ExprId) -> Option<ExprId> {
    let val = extract_integer_bigint(ctx, n)?;
    if val < BigInt::one() || !within_prime_search_cap(&val) {
        return None;
    }
    let mut result = val.clone();
    let mut remaining = val;
    let mut p = BigInt::from(2);
    while &p * &p <= remaining {
        if (&remaining % &p).is_zero() {
            while (&remaining % &p).is_zero() {
                remaining /= &p;
            }
            result = result / &p * (&p - 1);
        }
        p += 1;
    }
    if remaining > BigInt::one() {
        result = result / &remaining * (&remaining - 1);
    }
    Some(integer_result(ctx, result))
}

/// Compute integer GCD expression when both inputs are exact integers.
pub fn compute_integer_gcd_expr(ctx: &mut Context, a: ExprId, b: ExprId) -> Option<ExprId> {
    let val_a = extract_integer_bigint(ctx, a)?;
    let val_b = extract_integer_bigint(ctx, b)?;
    let gcd = val_a.gcd(&val_b);
    Some(ctx.add(Expr::Number(BigRational::from_integer(gcd))))
}

/// Compute `lcm(a, b)` when both inputs are exact integers.
pub fn compute_lcm_expr(ctx: &mut Context, a: ExprId, b: ExprId) -> Option<ExprId> {
    let val_a = extract_integer_bigint(ctx, a)?;
    let val_b = extract_integer_bigint(ctx, b)?;
    if val_a.is_zero() && val_b.is_zero() {
        return Some(ctx.num(0));
    }
    let lcm = val_a.lcm(&val_b);
    Some(ctx.add(Expr::Number(BigRational::from_integer(lcm))))
}

/// Compute Euclidean remainder `a mod n` when both inputs are exact integers.
pub fn compute_mod_expr(ctx: &mut Context, a: ExprId, n: ExprId) -> Option<ExprId> {
    let val_a = extract_integer_bigint(ctx, a)?;
    let val_n = extract_integer_bigint(ctx, n)?;
    if val_n.is_zero() {
        return None;
    }
    let rem = ((val_a % &val_n) + &val_n) % &val_n;
    Some(ctx.add(Expr::Number(BigRational::from_integer(rem))))
}

/// Compute prime factorization expression for exact integer input.
///
/// Returns a `factored(...)` expression with optional `factored_pow(base, exp)` nodes.
pub fn compute_prime_factors_expr(ctx: &mut Context, n: ExprId) -> Option<ExprId> {
    let val = extract_integer_bigint(ctx, n)?;
    if val.is_zero() {
        return Some(ctx.num(0));
    }
    if val.is_one() {
        return Some(ctx.num(1));
    }

    let sign = if val.is_negative() { -1 } else { 1 };
    let mut n_abs = val.abs();

    let mut factors = Vec::new();
    let one = BigInt::one();

    while n_abs.is_even() {
        factors.push(BigInt::from(2));
        n_abs /= 2;
    }

    let mut d = BigInt::from(3);
    while &d * &d <= n_abs {
        while (&n_abs % &d).is_zero() {
            factors.push(d.clone());
            n_abs /= &d;
        }
        d += 2;
    }
    if n_abs > one {
        factors.push(n_abs);
    }

    let mut grouped = Vec::new();
    if !factors.is_empty() {
        let mut current = factors[0].clone();
        let mut count = 1;
        for f in factors.iter().skip(1) {
            if *f == current {
                count += 1;
            } else {
                grouped.push((current, count));
                current = f.clone();
                count = 1;
            }
        }
        grouped.push((current, count));
    }

    let mut exprs = Vec::new();
    if sign == -1 {
        exprs.push(ctx.num(-1));
    }

    for (base, exp) in grouped {
        let base_expr = ctx.add(Expr::Number(BigRational::from_integer(base)));
        if exp == 1 {
            exprs.push(base_expr);
        } else {
            let exp_expr = ctx.num(exp);
            exprs.push(ctx.call("factored_pow", vec![base_expr, exp_expr]));
        }
    }

    if exprs.is_empty() {
        return Some(ctx.num(1));
    }

    Some(ctx.call("factored", exprs))
}

/// Compute `n!` for exact non-negative integer inputs, bounded by `n <= 1000`.
pub fn compute_factorial_expr(ctx: &mut Context, n: ExprId) -> Option<ExprId> {
    let val = extract_integer_bigint(ctx, n)?;
    if val.is_negative() {
        return None;
    }
    if val > BigInt::from(1000) {
        return None;
    }

    let mut res = BigInt::one();
    let mut i = BigInt::one();
    while i <= val {
        res *= &i;
        i += 1;
    }
    Some(ctx.add(Expr::Number(BigRational::from_integer(res))))
}

/// Compute binomial coefficient `n choose k` for exact integer inputs.
pub fn compute_choose_expr(ctx: &mut Context, n: ExprId, k: ExprId) -> Option<ExprId> {
    let val_n = extract_integer_bigint(ctx, n)?;
    let val_k = extract_integer_bigint(ctx, k)?;

    // `k < 0` is the empty selection under EVERY convention.
    if val_k.is_negative() {
        return Some(ctx.num(0));
    }
    // `k = 0 → 1` for ANY `n` — combinatorial empty selection AND the generalized
    // binomial. (Round-4 Cluster N: this boundary must be checked BEFORE the
    // `k > n → 0` short-circuit, which for `n < 0` would wrongly fabricate 0.)
    if val_k.is_zero() {
        return Some(ctx.num(1));
    }
    // For `n ≥ 0` the combinatorial rules apply: `k > n → 0`, `k == n → 1`, and the
    // symmetry `C(n,k) = C(n, n-k)` keeps the product short. For `n < 0` fall through
    // to the generalized binomial `C(n,k) = ff(n,k)/k!` (matches `sympy.binomial`,
    // e.g. `C(-1,2)=1`, `C(-2,3)=-4`) — never a fabricated 0.
    let k_eff = if !val_n.is_negative() {
        if val_k > val_n {
            return Some(ctx.num(0));
        }
        if val_k == val_n {
            return Some(ctx.num(1));
        }
        if &val_k * 2 > val_n {
            &val_n - &val_k
        } else {
            val_k.clone()
        }
    } else {
        val_k.clone()
    };

    let mut num = BigInt::one();
    let mut den = BigInt::one();
    let mut i = BigInt::zero();
    while i < k_eff {
        num *= &val_n - &i;
        den *= &i + 1;
        i += 1;
    }

    let res = num / den;
    Some(ctx.add(Expr::Number(BigRational::from_integer(res))))
}

/// Compute permutations `nPk` for exact integer inputs.
pub fn compute_perm_expr(ctx: &mut Context, n: ExprId, k: ExprId) -> Option<ExprId> {
    let val_n = extract_integer_bigint(ctx, n)?;
    let val_k = extract_integer_bigint(ctx, k)?;

    // `k < 0` is empty; `k = 0 → 1` for ANY `n` (checked BEFORE the `k > n → 0`
    // short-circuit so a negative `n` is not fabricated to 0 — Round-4 Cluster N).
    if val_k.is_negative() {
        return Some(ctx.num(0));
    }
    if val_k.is_zero() {
        return Some(ctx.num(1));
    }
    // `k > n → 0` is combinatorial and applies only for `n ≥ 0`. For `n < 0`,
    // `perm` continues as the falling factorial `ff(n,k)` (matches `sympy.ff`,
    // e.g. `perm(-3,2)=12`) — never a fabricated 0.
    if !val_n.is_negative() && val_k > val_n {
        return Some(ctx.num(0));
    }

    let mut res = BigInt::one();
    let mut i = BigInt::zero();
    while i < val_k {
        res *= &val_n - &i;
        i += 1;
    }

    Some(ctx.add(Expr::Number(BigRational::from_integer(res))))
}

/// Compute GCD for integers/polynomials.
///
/// Behavior:
/// - Integer inputs: Euclidean GCD (exact).
/// - Univariate polynomial inputs on same variable: polynomial Euclidean GCD.
/// - Multivariate or unsupported symbolic inputs: conservative fallback to `1`.
pub fn compute_gcd(ctx: &mut Context, a: ExprId, b: ExprId) -> Option<ExprId> {
    // Integer fast path.
    if extract_integer_bigint(ctx, a).is_some() && extract_integer_bigint(ctx, b).is_some() {
        return compute_integer_gcd_expr(ctx, a, b);
    }

    // Try univariate polynomial GCD.
    use crate::polynomial::Polynomial;

    let vars_a = cas_ast::collect_variables(ctx, a);
    let vars_b = cas_ast::collect_variables(ctx, b);

    if vars_a.len() == 1 && vars_a == vars_b {
        if let Some(var) = vars_a.iter().next() {
            if let (Ok(p_a), Ok(p_b)) = (
                Polynomial::from_expr(ctx, a, var),
                Polynomial::from_expr(ctx, b, var),
            ) {
                let gcd_poly = p_a.gcd(&p_b);
                return Some(gcd_poly.to_expr(ctx));
            }
        }
    }

    // Multivariable fallback: conservative, never incorrect.
    if !vars_a.is_empty() || !vars_b.is_empty() {
        return Some(ctx.num(1));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn choose_and_perm_handle_negative_n_via_generalized_binomial() {
        // Round-4 Cluster N: the `k > n -> 0` short-circuit must NOT fire before the
        // `k == 0 -> 1` boundary, and `n < 0` follows the generalized binomial
        // C(n,k) = ff(n,k)/k! (matches sympy.binomial) — never a fabricated 0.
        let int_val = |ctx: &Context, e: ExprId| -> i64 {
            match ctx.get(e) {
                Expr::Number(r) if r.is_integer() => {
                    use num_traits::ToPrimitive;
                    r.to_integer().to_i64().expect("fits i64")
                }
                other => panic!("expected integer, got {other:?}"),
            }
        };
        let mut ctx = Context::new();
        let chk_choose = |ctx: &mut Context, n: i64, k: i64, want: i64| {
            let (ne, ke) = (ctx.num(n), ctx.num(k));
            let got = compute_choose_expr(ctx, ne, ke).expect("choose");
            assert_eq!(int_val(ctx, got), want, "choose({n},{k})");
        };
        let chk_perm = |ctx: &mut Context, n: i64, k: i64, want: i64| {
            let (ne, ke) = (ctx.num(n), ctx.num(k));
            let got = compute_perm_expr(ctx, ne, ke).expect("perm");
            assert_eq!(int_val(ctx, got), want, "perm({n},{k})");
        };

        // The 5 audit probes + the generalized values (sympy-verified).
        chk_choose(&mut ctx, -5, 0, 1);
        chk_choose(&mut ctx, -1, 0, 1);
        chk_choose(&mut ctx, -1, 1, -1);
        chk_choose(&mut ctx, -1, 2, 1);
        chk_choose(&mut ctx, -1, 3, -1);
        chk_choose(&mut ctx, -2, 3, -4);
        chk_perm(&mut ctx, -3, 0, 1);
        chk_perm(&mut ctx, -3, 2, 12);

        // Positive controls unchanged + k>n -> 0 still holds for n>=0.
        chk_choose(&mut ctx, 5, 2, 10);
        chk_choose(&mut ctx, 5, 0, 1);
        chk_choose(&mut ctx, 5, 5, 1);
        chk_choose(&mut ctx, 2, 5, 0);
        chk_choose(&mut ctx, 0, 0, 1);
        chk_choose(&mut ctx, 10, 3, 120);
        chk_perm(&mut ctx, 5, 2, 20);
        chk_perm(&mut ctx, 3, 3, 6);
        chk_perm(&mut ctx, 2, 5, 0);

        // k < 0 is empty under every convention.
        chk_choose(&mut ctx, 5, -1, 0);
        chk_choose(&mut ctx, -1, -1, 0);
        chk_perm(&mut ctx, 5, -2, 0);
    }

    #[test]
    fn detects_poly_result_even_inside_hold() {
        let mut ctx = Context::new();
        let poly = parse("poly_result(7)", &mut ctx).expect("poly");
        let held = ctx.call_builtin(BuiltinFn::Hold, vec![poly]);
        let one = ctx.num(1);
        let wrapped = ctx.add(Expr::Add(one, held));
        assert!(contains_poly_result(&ctx, wrapped));
    }

    #[test]
    fn large_unexpanded_power_detection_matches_intent() {
        let mut ctx = Context::new();
        let large_compound = parse("(x+1)^3", &mut ctx).expect("compound");
        let large_atomic = parse("x^3", &mut ctx).expect("atomic");
        let poly_power = parse("(poly_result(3))^8", &mut ctx).expect("poly");

        assert!(has_large_unexpanded_power(&ctx, large_compound));
        assert!(!has_large_unexpanded_power(&ctx, large_atomic));
        assert!(has_large_unexpanded_power(&ctx, poly_power));
    }

    #[test]
    fn detects_sqrt_of_integer_in_pow_and_builtin_forms() {
        let mut ctx = Context::new();
        let five = ctx.num(5);
        let half = ctx.rational(1, 2);
        let pow_form = ctx.add(Expr::Pow(five, half));
        let sqrt_form = parse("sqrt(5)", &mut ctx).expect("sqrt");
        let wrong_value = parse("sqrt(3)", &mut ctx).expect("sqrt(3)");
        let symbolic = parse("sqrt(x)", &mut ctx).expect("sqrt(x)");

        assert!(is_sqrt_of_integer_expr(&ctx, pow_form, 5));
        assert!(is_sqrt_of_integer_expr(&ctx, sqrt_form, 5));
        assert!(!is_sqrt_of_integer_expr(&ctx, wrong_value, 5));
        assert!(!is_sqrt_of_integer_expr(&ctx, symbolic, 5));
    }

    #[test]
    fn integer_exponent_extraction_handles_negation() {
        let mut ctx = Context::new();
        let exp = parse("-5", &mut ctx).expect("exp");
        assert_eq!(get_integer_exponent(&ctx, exp), Some(-5));
    }

    #[test]
    fn computes_integer_number_theory_primitives() {
        let mut ctx = Context::new();
        let a = ctx.num(12);
        let b = ctx.num(18);
        let n = ctx.num(5);
        let two = ctx.num(2);
        let minus_seven = ctx.num(-7);
        let five = ctx.num(5);

        let lcm = compute_lcm_expr(&mut ctx, a, b).expect("lcm");
        assert_eq!(extract_integer_bigint(&ctx, lcm), Some(BigInt::from(36)));

        let gcd = compute_integer_gcd_expr(&mut ctx, a, b).expect("gcd");
        assert_eq!(extract_integer_bigint(&ctx, gcd), Some(BigInt::from(6)));

        let modu = compute_mod_expr(&mut ctx, minus_seven, five).expect("mod");
        assert_eq!(extract_integer_bigint(&ctx, modu), Some(BigInt::from(3)));

        let fact = compute_factorial_expr(&mut ctx, n).expect("fact");
        assert_eq!(extract_integer_bigint(&ctx, fact), Some(BigInt::from(120)));

        let choose = compute_choose_expr(&mut ctx, five, two).expect("choose");
        assert_eq!(extract_integer_bigint(&ctx, choose), Some(BigInt::from(10)));

        let perm = compute_perm_expr(&mut ctx, five, two).expect("perm");
        assert_eq!(extract_integer_bigint(&ctx, perm), Some(BigInt::from(20)));
    }

    #[test]
    fn folds_gcd_and_lcm_over_three_or_more_arguments() {
        // Regression: gcd/lcm of 3+ args must fold over ALL of them, not just the first two
        // (gcd(8, 12, 6) silently returned 4 instead of 2 before the fix).
        for (name, values, expected) in [
            ("gcd", vec![8, 12, 6], 2),
            ("gcd", vec![30, 20, 12], 2),
            ("gcd", vec![100, 60, 45], 5),
            ("gcd", vec![7, 11, 13], 1),
            ("lcm", vec![4, 6, 8], 24),
            ("lcm", vec![2, 3, 4, 5], 60),
        ] {
            let mut ctx = Context::new();
            let args: Vec<ExprId> = values.iter().map(|&v| ctx.num(v)).collect();
            let rewrite = try_eval_simple_number_theory_call(&mut ctx, name, &args)
                .unwrap_or_else(|| panic!("{name}{values:?} should evaluate"));
            assert_eq!(
                extract_integer_bigint(&ctx, rewrite.result()),
                Some(BigInt::from(expected)),
                "{name}{values:?}"
            );
        }
    }

    #[test]
    fn selects_poly_gcd_mode_explicit_or_auto() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x2 = ctx.add(Expr::Pow(x, two));
        let x3 = ctx.add(Expr::Pow(x, three));

        // Explicit mode always wins
        let explicit = select_poly_gcd_mode(&ctx, x2, x3, Some(GcdMode::Structural));
        assert_eq!(explicit, GcdMode::Structural);

        // Auto mode: large unexpanded power over non-atomic base forces Modp
        let one = ctx.num(1);
        let five = ctx.num(5);
        let non_atomic_base = ctx.add(Expr::Add(x, one));
        let big_pow = ctx.add(Expr::Pow(non_atomic_base, five));
        let auto_modp = select_poly_gcd_mode(&ctx, big_pow, x2, None);
        assert_eq!(auto_modp, GcdMode::Modp);

        // Auto mode: small/simple cases stay structural
        let auto_structural = select_poly_gcd_mode(&ctx, x2, x3, None);
        assert_eq!(auto_structural, GcdMode::Structural);
    }

    #[test]
    fn compute_gcd_returns_integer_result() {
        let mut ctx = Context::new();
        let a = ctx.num(48);
        let b = ctx.num(18);
        let gcd = compute_gcd(&mut ctx, a, b).expect("gcd");
        assert_eq!(extract_integer_bigint(&ctx, gcd), Some(BigInt::from(6)));
    }

    #[test]
    fn evaluates_simple_named_number_theory_calls() {
        let mut ctx = Context::new();
        let a = ctx.num(12);
        let b = ctx.num(18);
        let five = ctx.num(5);
        let two = ctx.num(2);

        let gcd_call =
            try_eval_simple_number_theory_call(&mut ctx, "gcd", &[a, b]).expect("gcd call");
        assert_eq!(gcd_call.name(), "gcd");
        assert_eq!(
            extract_integer_bigint(&ctx, gcd_call.result()),
            Some(BigInt::from(6))
        );

        let choose_call =
            try_eval_simple_number_theory_call(&mut ctx, "nCr", &[five, two]).expect("choose call");
        assert_eq!(choose_call.name(), "choose");
        assert_eq!(
            extract_integer_bigint(&ctx, choose_call.result()),
            Some(BigInt::from(10))
        );

        let factor_call =
            try_eval_simple_number_theory_call(&mut ctx, "factorial", &[five]).expect("fact");
        assert_eq!(factor_call.name(), "fact");
        assert_eq!(
            extract_integer_bigint(&ctx, factor_call.result()),
            Some(BigInt::from(120))
        );

        let x = ctx.var("x");
        assert!(try_eval_simple_number_theory_call(&mut ctx, "gcd", &[x, five]).is_none());
    }

    #[test]
    fn rewrites_numeric_pascal_choose_identity() {
        let mut ctx = Context::new();
        let expr = parse("choose(4,1) + choose(4,2)", &mut ctx).expect("pascal");
        let expected = parse("choose(5,2)", &mut ctx).expect("expected");

        let rewrite =
            try_rewrite_pascal_choose_identity_expr(&mut ctx, expr).expect("pascal rewrite");

        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn pascal_choose_identity_rejects_nonconsecutive_terms() {
        let mut ctx = Context::new();
        let expr = parse("choose(4,1) + choose(4,3)", &mut ctx).expect("not pascal");

        assert!(try_rewrite_pascal_choose_identity_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn rewrites_numeric_choose_symmetry_to_complement_target() {
        let mut ctx = Context::new();
        let expr = parse("choose(6,1)", &mut ctx).expect("choose symmetry");
        let expected = parse("choose(6,5)", &mut ctx).expect("expected");

        let rewrite = try_rewrite_choose_symmetry_expr(&mut ctx, expr).expect("symmetry rewrite");

        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn choose_symmetry_rejects_reverse_orientation() {
        let mut ctx = Context::new();
        let expr = parse("choose(6,5)", &mut ctx).expect("reverse symmetry");

        assert!(try_rewrite_choose_symmetry_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn dispatches_simple_and_poly_gcd_paths() {
        let mut ctx = Context::new();

        let gcd_int = parse("gcd(12, 18)", &mut ctx).expect("gcd int");
        match dispatch_number_theory_call(&mut ctx, gcd_int).expect("dispatch int") {
            NumberTheoryDispatch::Simple(call) => {
                assert_eq!(call.name(), "gcd");
                assert_eq!(
                    extract_integer_bigint(&ctx, call.result()),
                    Some(BigInt::from(6))
                );
            }
            NumberTheoryDispatch::PolyGcd { .. } => panic!("expected simple gcd path"),
        }

        let gcd_poly = parse("gcd(x^2-1, x-1)", &mut ctx).expect("gcd poly");
        match dispatch_number_theory_call(&mut ctx, gcd_poly).expect("dispatch poly") {
            NumberTheoryDispatch::PolyGcd { lhs, rhs, .. } => {
                assert_eq!(lhs, parse("x^2 - 1", &mut ctx).expect("lhs"));
                assert_eq!(rhs, parse("x - 1", &mut ctx).expect("rhs"));
            }
            NumberTheoryDispatch::Simple(_) => panic!("expected poly gcd fallback"),
        }
    }

    #[test]
    fn rewrites_consecutive_factorial_ratio() {
        let mut ctx = Context::new();
        let expr = parse("(n + 1)! / n!", &mut ctx).expect("parse");
        let expected = parse("n + 1", &mut ctx).expect("expected");
        let expected_arg = parse("n", &mut ctx).expect("arg");

        let rewrite =
            try_rewrite_consecutive_factorial_ratio_expr(&mut ctx, expr).expect("rewrite");

        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            compare_expr(
                &ctx,
                rewrite.factorial_arg_requires_nonnegative,
                expected_arg
            ),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn rewrites_named_factorial_ratio() {
        let mut ctx = Context::new();
        let expr = parse("factorial(1 + n) / factorial(n)", &mut ctx).expect("parse");
        let expected = parse("n + 1", &mut ctx).expect("expected");

        let rewrite =
            try_rewrite_consecutive_factorial_ratio_expr(&mut ctx, expr).expect("rewrite");

        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn rewrites_factorial_ratio_with_gap_two() {
        let mut ctx = Context::new();
        let expr = parse("(n + 1)! / (n - 1)!", &mut ctx).expect("parse");
        let expected = parse("n * (n + 1)", &mut ctx).expect("expected");
        let expected_arg = parse("n - 1", &mut ctx).expect("arg");

        let rewrite =
            try_rewrite_consecutive_factorial_ratio_expr(&mut ctx, expr).expect("rewrite");

        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            compare_expr(
                &ctx,
                rewrite.factorial_arg_requires_nonnegative,
                expected_arg
            ),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn does_not_rewrite_unrelated_factorial_ratio() {
        let mut ctx = Context::new();
        let expr = parse("(n + 1)! / m!", &mut ctx).expect("parse");
        assert!(try_rewrite_consecutive_factorial_ratio_expr(&mut ctx, expr).is_none());
    }
}
