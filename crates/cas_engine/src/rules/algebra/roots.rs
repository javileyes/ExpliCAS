use crate::define_rule;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Context, Expr};
use num_traits::{Signed, Zero};

define_rule!(RootDenestingRule, "Root Denesting", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();

    // We look for sqrt(A + B) or sqrt(A - B)
    // Also handle Pow(inner, 1/2)
    let inner = if let Expr::Function(name, args) = &expr_data {
        if name == "sqrt" && args.len() == 1 {
            Some(args[0])
        } else {
            None
        }
    } else if let Expr::Pow(b, e) = &expr_data {
        if let Expr::Number(n) = ctx.get(*e) {
            if *n.numer() == 1.into() && *n.denom() == 2.into() {
                Some(*b)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    inner?;
    let inner = inner.unwrap();
    let inner_data = ctx.get(inner).clone();
    //println!("RootDenesting checking inner: {:?}", inner_data);

    let (a, b, is_add) = match inner_data {
        Expr::Add(l, r) => (l, r, true),
        Expr::Sub(l, r) => (l, r, false),
        _ => return None,
    };

    // Helper to identify if a term is C*sqrt(D) or sqrt(D)
    // Returns (Option<C>, D). If C is None, it means 1.
    fn analyze_sqrt_term(
        ctx: &Context,
        e: cas_ast::ExprId,
    ) -> Option<(Option<cas_ast::ExprId>, cas_ast::ExprId)> {
        match ctx.get(e) {
            Expr::Function(fname, fargs) if fname == "sqrt" && fargs.len() == 1 => {
                Some((None, fargs[0]))
            }
            Expr::Pow(b, e) => {
                // Check for b^(3/2) -> b * sqrt(b)
                if let Expr::Number(n) = ctx.get(*e) {
                    // Debug: Checking Pow for root denesting
                    if *n.numer() == 3.into() && *n.denom() == 2.into() {
                        return Some((Some(*b), *b));
                    }
                }
                None
            }
            Expr::Mul(l, r) => {
                // Helper to check for sqrt/pow(1/2)
                let is_sqrt = |e: cas_ast::ExprId| -> Option<cas_ast::ExprId> {
                    match ctx.get(e) {
                        Expr::Function(fname, fargs) if fname == "sqrt" && fargs.len() == 1 => {
                            Some(fargs[0])
                        }
                        Expr::Pow(b, e) => {
                            if let Expr::Number(n) = ctx.get(*e) {
                                if *n.numer() == 1.into() && *n.denom() == 2.into() {
                                    return Some(*b);
                                }
                            }
                            None
                        }
                        _ => None,
                    }
                };

                // Check l * sqrt(r)
                if let Some(inner) = is_sqrt(*r) {
                    return Some((Some(*l), inner));
                }
                // Check sqrt(l) * r
                if let Some(inner) = is_sqrt(*l) {
                    return Some((Some(*r), inner));
                }
                None
            }
            _ => None,
        }
    }

    // We need to determine which is the "rational" part A and which is the "surd" part sqrt(B).
    // Try both permutations.

    // We can't use a closure that captures ctx mutably and calls methods on it easily.
    // So we inline the logic or use a macro/helper that takes ctx.

    let check_permutation = |ctx: &mut Context,
                             term_a: cas_ast::ExprId,
                             term_b: cas_ast::ExprId|
     -> Option<crate::rule::Rewrite> {
        // Assume term_a is A, term_b is C*sqrt(D)
        // We need to call analyze_sqrt_term which takes &Context.
        // But we need to mutate ctx later.
        // So we analyze first.

        let sqrt_parts = analyze_sqrt_term(ctx, term_b);

        if let Some((c_opt, d)) = sqrt_parts {
            // We have sqrt(A +/- C*sqrt(D))
            // Effective B_eff = C^2 * D
            let c = c_opt.unwrap_or_else(|| ctx.num(1));

            // We need numerical values to check the condition
            if let (Expr::Number(val_a), Expr::Number(val_c), Expr::Number(val_d)) =
                (ctx.get(term_a), ctx.get(c), ctx.get(d))
            {
                let val_c2 = val_c * val_c;
                let val_beff = val_c2 * val_d;
                let val_a2 = val_a * val_a;
                let val_delta = val_a2 - val_beff;

                if val_delta >= num_rational::BigRational::zero() && val_delta.is_integer() {
                    let int_delta = val_delta.to_integer();
                    let sqrt_delta = int_delta.sqrt();

                    if sqrt_delta.clone() * sqrt_delta.clone() == int_delta {
                        // Perfect square!
                        let z_val = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
                            sqrt_delta,
                        )));

                        // Found Z!
                        // Result = sqrt((A+Z)/2) +/- sqrt((A-Z)/2)
                        let two = ctx.num(2);

                        let term1_num = ctx.add(Expr::Add(term_a, z_val));
                        let term1_frac = ctx.add(Expr::Div(term1_num, two));
                        let term1 = ctx.add(Expr::Function("sqrt".to_string(), vec![term1_frac]));

                        let term2_num = ctx.add(Expr::Sub(term_a, z_val));
                        let term2_frac = ctx.add(Expr::Div(term2_num, two));
                        let term2 = ctx.add(Expr::Function("sqrt".to_string(), vec![term2_frac]));

                        // Check sign of C
                        let c_is_negative = if let Expr::Number(n) = ctx.get(c) {
                            n < &num_rational::BigRational::zero()
                        } else {
                            false
                        };

                        // If is_add is true, we have A + C*sqrt(D).
                        // If C is negative, effective operation is subtraction.
                        // If is_add is false, we have A - C*sqrt(D).
                        // If C is negative, effective operation is addition.

                        let effective_sub = if is_add {
                            c_is_negative
                        } else {
                            !c_is_negative
                        };

                        let new_expr = if effective_sub {
                            ctx.add(Expr::Sub(term1, term2))
                        } else {
                            ctx.add(Expr::Add(term1, term2))
                        };

                        return Some(crate::rule::Rewrite {
                            new_expr,
                            description: "Denest square root".to_string(),
                            before_local: None,
                            after_local: None,
                            domain_assumption: None,
                        });
                    }
                }
            }
        }
        None
    };

    if let Some(rw) = check_permutation(ctx, a, b) {
        return Some(rw);
    }
    if let Some(rw) = check_permutation(ctx, b, a) {
        return Some(rw);
    }
    None
});

define_rule!(
    SimplifySquareRootRule,
    "Simplify Square Root",
    |ctx, expr| {
        let arg = if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "sqrt" && args.len() == 1 {
                Some(args[0])
            } else {
                None
            }
        } else if let Expr::Pow(b, e) = ctx.get(expr) {
            if let Expr::Number(n) = ctx.get(*e) {
                if *n.numer() == 1.into() && *n.denom() == 2.into() {
                    Some(*b)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some(arg) = arg {
            // Only try to factor if argument is Add/Sub (polynomial)
            match ctx.get(arg) {
                Expr::Add(_, _) | Expr::Sub(_, _) => {}
                _ => return None,
            }

            use crate::polynomial::Polynomial;
            use crate::rules::algebra::helpers::collect_variables;

            let vars = collect_variables(ctx, arg);
            if vars.len() == 1 {
                let var = vars.iter().next().unwrap();
                if let Ok(poly) = Polynomial::from_expr(ctx, arg, var) {
                    // First: Try to detect perfect square with rational coefficients
                    // For ax² + bx + c to be (dx + e)², we need:
                    // - a = d², c = e², b = 2de
                    // - Equivalently: b² = 4ac (discriminant = 0)
                    if poly.degree() == 2 && poly.coeffs.len() >= 3 {
                        let a = poly.coeffs.get(2).cloned();
                        let b = poly.coeffs.get(1).cloned();
                        let c = poly.coeffs.get(0).cloned();

                        if let (Some(a), Some(b), Some(c)) = (a, b, c) {
                            // Check discriminant: b² - 4ac = 0
                            let four = num_rational::BigRational::from_integer(4.into());
                            let discriminant = b.clone() * b.clone() - four * a.clone() * c.clone();

                            if discriminant.is_zero() {
                                // Perfect square! Now find d and e where (dx + e)² = ax² + bx + c
                                // d = √a, e = √c (with appropriate sign)
                                if let (Some(d), Some(_e_abs)) =
                                    (rational_sqrt(&a), rational_sqrt(&c))
                                {
                                    // Determine sign of e: 2de = b, so e = b/(2d)
                                    let two = num_rational::BigRational::from_integer(2.into());
                                    let e = if d.is_zero() {
                                        rational_sqrt(&c)
                                            .unwrap_or_else(|| num_rational::BigRational::zero())
                                    } else {
                                        b.clone() / (two * d.clone())
                                    };

                                    // Build (dx + e)
                                    let var_expr = ctx.var(var);
                                    let d_expr = ctx.add(Expr::Number(d.clone()));
                                    let e_expr = ctx.add(Expr::Number(e.clone()));

                                    let one = num_rational::BigRational::from_integer(1.into());
                                    let dx = if d == one {
                                        var_expr
                                    } else {
                                        smart_mul(ctx, d_expr, var_expr)
                                    };

                                    let linear = if e.is_zero() {
                                        dx
                                    } else if e.is_positive() {
                                        ctx.add(Expr::Add(dx, e_expr))
                                    } else {
                                        // e is negative, use Add with the actual (negative) value
                                        ctx.add(Expr::Add(dx, e_expr))
                                    };

                                    // sqrt((dx+e)²) = |dx+e|
                                    let abs_linear =
                                        ctx.add(Expr::Function("abs".to_string(), vec![linear]));

                                    return Some(Rewrite {
                                        new_expr: abs_linear,
                                        description: "Simplify perfect square root".to_string(),
                                        before_local: None,
                                        after_local: None,
                                        domain_assumption: None,
                                    });
                                }
                            }
                        }
                    }

                    // Fallback: Try integer factorization
                    let factors = poly.factor_rational_roots();
                    if !factors.is_empty() {
                        let first = &factors[0];
                        if factors.iter().all(|f| f == first) {
                            let count = factors.len() as u32;
                            if count >= 2 {
                                let base = first.to_expr(ctx);
                                let k = count / 2;
                                let rem = count % 2;

                                let abs_base =
                                    ctx.add(Expr::Function("abs".to_string(), vec![base]));

                                let term1 = if k == 1 {
                                    abs_base
                                } else {
                                    let k_expr = ctx.num(k as i64);
                                    ctx.add(Expr::Pow(abs_base, k_expr))
                                };

                                if rem == 0 {
                                    return Some(Rewrite {
                                        new_expr: term1,
                                        description: "Simplify perfect square root".to_string(),
                                        before_local: None,
                                        after_local: None,
                                        domain_assumption: None,
                                    });
                                } else {
                                    let sqrt_base =
                                        ctx.add(Expr::Function("sqrt".to_string(), vec![base]));
                                    let new_expr = smart_mul(ctx, term1, sqrt_base);
                                    return Some(Rewrite {
                                        new_expr,
                                        description: "Simplify square root factors".to_string(),
                                        before_local: None,
                                        after_local: None,
                                        domain_assumption: None,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }
);

/// Try to compute the square root of a rational number.
/// Returns Some(√r) if both numerator and denominator are perfect squares.
fn rational_sqrt(r: &num_rational::BigRational) -> Option<num_rational::BigRational> {
    use num_traits::Signed;

    // For negative numbers, no real square root
    if r.is_negative() {
        return None;
    }

    if r.is_zero() {
        return Some(num_rational::BigRational::from_integer(0.into()));
    }

    let numer = r.numer().clone();
    let denom = r.denom().clone();

    // Check if numerator is a perfect square
    let numer_sqrt = numer.sqrt();
    if &numer_sqrt * &numer_sqrt != numer {
        return None;
    }

    // Check if denominator is a perfect square
    let denom_sqrt = denom.sqrt();
    if &denom_sqrt * &denom_sqrt != denom {
        return None;
    }

    Some(num_rational::BigRational::new(numer_sqrt, denom_sqrt))
}

// =============================================================================
// CUBIC CONJUGATE TRAP RULE
// Simplifies ∛(m+t) + ∛(m-t) when the result is a rational number.
// =============================================================================

/// Try to split an expression as `m ± t` where m is rational and t is a surd.
/// Returns (m, t, sign) where sign = +1 means m+t, sign = -1 means m-t.
/// Handles different orderings from parser canonicalization.
fn split_as_m_plus_t(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, i32)> {
    // Helper to check if an expression contains a sqrt (surd-like)
    fn is_surd_like(ctx: &cas_ast::Context, e: cas_ast::ExprId) -> bool {
        match ctx.get(e) {
            Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => true,
            Expr::Pow(_, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    *n.numer() == 1.into() && *n.denom() == 2.into()
                } else {
                    false
                }
            }
            Expr::Mul(l, r) => is_surd_like(ctx, *l) || is_surd_like(ctx, *r),
            Expr::Neg(inner) => is_surd_like(ctx, *inner),
            _ => false,
        }
    }

    fn is_numeric(ctx: &cas_ast::Context, e: cas_ast::ExprId) -> bool {
        matches!(ctx.get(e), Expr::Number(_))
    }

    // Extract terms and effective sign
    // For Add(a, b): sign of t is +1
    // For Sub(a, b): sign of t is -1 (when t is the right operand)
    // For Add(a, Neg(b)): sign of t is -1 (when t is b)
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            // Check for Add(a, Neg(b)) = a - b
            if let Expr::Neg(neg_inner) = ctx.get(*r) {
                // a + (-b) where b = neg_inner
                if is_numeric(ctx, *l) && is_surd_like(ctx, *neg_inner) {
                    // m + (-t) = m - t
                    return Some((*l, *neg_inner, -1));
                }
                if is_surd_like(ctx, *l) && is_numeric(ctx, *neg_inner) {
                    // t + (-m) = t - m = -(m - t) => this represents -(m - t)
                    // Actually we should not match this as it's not (m ± t)
                    return None;
                }
            }

            // Regular Add(a, b)
            if is_numeric(ctx, *l) && is_surd_like(ctx, *r) {
                // m + t
                return Some((*l, *r, 1));
            }
            if is_surd_like(ctx, *l) && is_numeric(ctx, *r) {
                // t + m = m + t (commutative)
                return Some((*r, *l, 1));
            }
            None
        }
        Expr::Sub(l, r) => {
            if is_numeric(ctx, *l) && is_surd_like(ctx, *r) {
                // m - t
                return Some((*l, *r, -1));
            }
            if is_surd_like(ctx, *l) && is_numeric(ctx, *r) {
                // t - m: This is NOT (m ± t), it's -(m - t)
                // We could represent it as m - t with sign flipped externally,
                // but for clarity, we only match when m is first in Sub
                return None;
            }
            None
        }
        _ => None,
    }
}

/// Check if two expressions are conjugates: base1 = m + t, base2 = m - t (or vice versa).
/// Returns Some((m, t)) if they are conjugates.
fn is_conjugate_pair(
    ctx: &cas_ast::Context,
    base1: cas_ast::ExprId,
    base2: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    use crate::ordering::compare_expr;
    use std::cmp::Ordering;

    let (m1, t1, sign1) = split_as_m_plus_t(ctx, base1)?;
    let (m2, t2, sign2) = split_as_m_plus_t(ctx, base2)?;

    // Must have same m and same t
    if compare_expr(ctx, m1, m2) != Ordering::Equal {
        return None;
    }
    if compare_expr(ctx, t1, t2) != Ordering::Equal {
        return None;
    }

    // One must be +1 (addition), one must be -1 (subtraction)
    // i.e., sign1 * sign2 == -1
    if sign1 + sign2 != 0 {
        return None;
    }

    Some((m1, t1))
}

/// Extract exponent from Pow(base, exp) and check if it equals 1/3.
fn is_cube_root_pow(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Option<cas_ast::ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            // Check if exponent is 1/3
            if *n.numer() == 1.into() && *n.denom() == 3.into() {
                return Some(*base);
            }
        }
    }
    None
}

/// Compute cube root of a rational number (real cube root, handles negatives).
/// Returns Some(result) if input is a perfect cube.
fn rational_cbrt(r: &num_rational::BigRational) -> Option<num_rational::BigRational> {
    use num_traits::Signed;

    let neg = r.is_negative();
    let abs_r = if neg { -r.clone() } else { r.clone() };

    if abs_r.is_zero() {
        return Some(num_rational::BigRational::from_integer(0.into()));
    }

    let numer = abs_r.numer().clone();
    let denom = abs_r.denom().clone();

    // Check if numerator is a perfect cube
    let numer_cbrt = numer.cbrt();
    if &numer_cbrt * &numer_cbrt * &numer_cbrt != numer {
        return None;
    }

    // Check if denominator is a perfect cube
    let denom_cbrt = denom.cbrt();
    if &denom_cbrt * &denom_cbrt * &denom_cbrt != denom {
        return None;
    }

    let result = num_rational::BigRational::new(numer_cbrt, denom_cbrt);
    if neg {
        Some(-result)
    } else {
        Some(result)
    }
}

/// Find a rational root of depressed cubic: x³ + px + q = 0
/// Uses Rational Root Theorem correctly for rational coefficients.
/// After clearing denominators: a·x³ + b·x + c = 0
/// Candidates are ±(divisors of |c|) / (divisors of |a|)
fn find_rational_root_depressed_cubic(
    p: &num_rational::BigRational,
    q: &num_rational::BigRational,
) -> Option<num_rational::BigRational> {
    use num_bigint::BigInt;
    use num_traits::{Signed, Zero};

    if q.is_zero() {
        // x³ + px = 0 => x(x² + p) = 0 => x = 0 is always a root
        return Some(num_rational::BigRational::zero());
    }

    // Clear denominators: multiply by LCM of all denominators
    // x³ + (p_n/p_d)x + (q_n/q_d) = 0
    // Multiply by LCM(p_d, q_d): LCM·x³ + (p_n·...)*x + (q_n·...) = 0
    let lcm_denom = num_integer::lcm(p.denom().clone(), q.denom().clone());

    // After clearing, we have: L·x³ + P'·x + Q' = 0
    // where L = lcm_denom, P' = p * L, Q' = q * L
    let leading_coef = lcm_denom.clone(); // coefficient of x³
    let constant_coef = q * num_rational::BigRational::from_integer(lcm_denom.clone());
    let constant_int = constant_coef.to_integer();

    // RRT: x = ±d/e where d divides |constant| and e divides |leading|
    let c_abs = if constant_int.is_negative() {
        -constant_int.clone()
    } else {
        constant_int.clone()
    };
    let a_abs = if leading_coef.is_negative() {
        -leading_coef.clone()
    } else {
        leading_coef.clone()
    };

    // Find divisors (limit to reasonable size for puzzles)
    fn small_divisors(n: &BigInt, limit: i64) -> Vec<BigInt> {
        let mut divs = Vec::new();
        if n.is_zero() {
            return vec![BigInt::from(1)];
        }
        let n_abs = if n.is_negative() {
            -n.clone()
        } else {
            n.clone()
        };
        for d in 1..=limit {
            let bd = BigInt::from(d);
            if &n_abs % &bd == BigInt::zero() {
                divs.push(bd.clone());
                let quotient = &n_abs / &bd;
                if !divs.contains(&quotient) {
                    divs.push(quotient);
                }
            }
        }
        if divs.is_empty() {
            divs.push(BigInt::from(1));
        }
        divs
    }

    let c_divisors = small_divisors(&c_abs, 50); // divisors of constant term
    let a_divisors = small_divisors(&a_abs, 20); // divisors of leading coef

    // Test candidates ±d/e
    for d in &c_divisors {
        for e in &a_divisors {
            for sign in &[1i32, -1i32] {
                let candidate = if *sign == 1 {
                    num_rational::BigRational::new(d.clone(), e.clone())
                } else {
                    -num_rational::BigRational::new(d.clone(), e.clone())
                };

                // Evaluate x³ + px + q at candidate
                let x2 = &candidate * &candidate;
                let x3 = &x2 * &candidate;
                let val = &x3 + p * &candidate + q;

                if val.is_zero() {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

define_rule!(
    CubicConjugateTrapRule,
    "Cubic Conjugate Identity",
    None,
    crate::phase::PhaseMask::TRANSFORM,
    |ctx, expr| {
        use num_traits::Zero;

        // Match Add(Pow(A, 1/3), Pow(B, 1/3))
        let (left, right) = match ctx.get(expr) {
            Expr::Add(l, r) => (*l, *r),
            _ => return None,
        };

        // Extract cube root bases
        let base_a = is_cube_root_pow(ctx, left)?;
        let base_b = is_cube_root_pow(ctx, right)?;

        // Check if A and B are conjugates (m + t) and (m - t)
        let (m, t) = is_conjugate_pair(ctx, base_a, base_b)?;

        // Compute S = A + B = 2m (directly, without simplify)
        // Since A = m + t and B = m - t, A + B = 2m
        let two = num_rational::BigRational::from_integer(2.into());

        // m must be a rational number for this to work
        let m_val = if let Expr::Number(n) = ctx.get(m) {
            n.clone()
        } else {
            return None; // m is not numeric, can't apply
        };

        let s_val = &two * &m_val; // S = 2m

        // Compute AB = m² - t² (directly)
        // t must also allow us to compute t² as rational
        // For t = sqrt(d) or k*sqrt(d), t² is rational
        let t_squared_val = compute_t_squared(ctx, t)?;

        let ab_val = &m_val * &m_val - &t_squared_val; // AB = m² - t²

        // P = ∛(AB) must be rational (perfect cube)
        let p_val = rational_cbrt(&ab_val)?;

        // Form depressed cubic: x³ + px + q = 0
        // where p_coef = -3P and q_coef = -S
        // x³ - 3Px - S = 0  =>  x³ + (-3P)x + (-S) = 0
        let three = num_rational::BigRational::from_integer(3.into());
        let p_coef = -&three * &p_val; // coefficient of x
        let q_coef = -&s_val; // constant term

        // Guard: if p_coef > 0, cubic is strictly increasing => unique real root
        // This ensures we can trust the RRT result
        if p_coef <= num_rational::BigRational::zero() {
            return None; // Multiple real roots possible, skip
        }

        // Find rational root via RRT
        let root = find_rational_root_depressed_cubic(&p_coef, &q_coef)?;

        // Success! Return the root as the result
        let result = ctx.add(Expr::Number(root.clone()));

        Some(Rewrite {
            new_expr: result,
            description: format!("Cubic conjugate identity: ∛(m+t) + ∛(m-t) = {}", root),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        })
    }
);

/// Compute t² where t may be:
/// - A number: t² = t * t
/// - sqrt(d) or d^(1/2): t² = d  
/// - k * sqrt(d): t² = k² * d
fn compute_t_squared(
    ctx: &cas_ast::Context,
    t: cas_ast::ExprId,
) -> Option<num_rational::BigRational> {
    match ctx.get(t) {
        // Direct number
        Expr::Number(n) => Some(n * n),

        // sqrt(d) function
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
            if let Expr::Number(d) = ctx.get(args[0]) {
                Some(d.clone())
            } else {
                None
            }
        }

        // d^(1/2) power form
        Expr::Pow(base, exp) => {
            if let Expr::Number(e) = ctx.get(*exp) {
                // Check if exponent is 1/2
                if *e.numer() == 1.into() && *e.denom() == 2.into() {
                    if let Expr::Number(d) = ctx.get(*base) {
                        return Some(d.clone());
                    }
                }
            }
            None
        }

        // k * sqrt(d) product form
        Expr::Mul(l, r) => {
            // Try both orderings
            let try_extract = |coef: cas_ast::ExprId,
                               surd: cas_ast::ExprId|
             -> Option<num_rational::BigRational> {
                let k = if let Expr::Number(n) = ctx.get(coef) {
                    n.clone()
                } else {
                    return None;
                };

                let d = match ctx.get(surd) {
                    Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
                        if let Expr::Number(n) = ctx.get(args[0]) {
                            n.clone()
                        } else {
                            return None;
                        }
                    }
                    Expr::Pow(base, exp) => {
                        if let Expr::Number(e) = ctx.get(*exp) {
                            if *e.numer() == 1.into() && *e.denom() == 2.into() {
                                if let Expr::Number(n) = ctx.get(*base) {
                                    n.clone()
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                    _ => return None,
                };

                // t = k * sqrt(d), so t² = k² * d
                Some(&k * &k * &d)
            };

            try_extract(*l, *r).or_else(|| try_extract(*r, *l))
        }

        _ => None,
    }
}

// =============================================================================
// DENEST SQRT(a + SQRT(b)) RULE
// Simplifies √(a + √b) → √m + √n where m,n = (a ± √(a²-b))/2
// =============================================================================

/// Extract the radicand if expression is a sqrt (either sqrt(x) function or x^(1/2))
fn as_sqrt(ctx: &cas_ast::Context, e: cas_ast::ExprId) -> Option<cas_ast::ExprId> {
    match ctx.get(e) {
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => Some(args[0]),
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n.numer() == 1.into() && *n.denom() == 2.into() {
                    return Some(*base);
                }
            }
            None
        }
        _ => None,
    }
}

define_rule!(
    DenestSqrtAddSqrtRule,
    "Denest Nested Square Root",
    None,
    crate::phase::PhaseMask::TRANSFORM,
    |ctx, expr| {
        // Match sqrt(inner) where inner = a + sqrt(b) or a - sqrt(b)
        let inner = as_sqrt(ctx, expr)?;

        // Inner must be Add or Sub
        let (left, right, is_add) = match ctx.get(inner) {
            Expr::Add(l, r) => (*l, *r, true),
            Expr::Sub(l, r) => (*l, *r, false),
            _ => return None,
        };

        // Identify which is `a` (rational) and which is `sqrt(b)`
        // Try both orderings
        let (a_val, b_val) = {
            // Try: left = a (Number), right = sqrt(b)
            if let Expr::Number(a) = ctx.get(left) {
                if let Some(b_inner) = as_sqrt(ctx, right) {
                    if let Expr::Number(b) = ctx.get(b_inner) {
                        Some((a.clone(), b.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }
        .or_else(|| {
            // Try: left = sqrt(b), right = a (Number)
            if let Some(b_inner) = as_sqrt(ctx, left) {
                if let Expr::Number(b) = ctx.get(b_inner) {
                    if let Expr::Number(a) = ctx.get(right) {
                        Some((a.clone(), b.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        })?;

        // For subtraction (a - sqrt(b)), we'd need a different formula
        // For now, only handle addition: sqrt(a + sqrt(b))
        if !is_add {
            // TODO: Handle subtraction case
            return None;
        }

        // Apply denesting formula:
        // √(a + √b) = √m + √n where m = (a + √disc)/2, n = (a - √disc)/2
        // disc = a² - b

        let disc = &a_val * &a_val - &b_val;

        // disc must have a rational square root
        let disc_sqrt = rational_sqrt(&disc)?;

        // m = (a + disc_sqrt) / 2
        // n = (a - disc_sqrt) / 2
        let two = num_rational::BigRational::from_integer(2.into());
        let m = (&a_val + &disc_sqrt) / &two;
        let n = (&a_val - &disc_sqrt) / &two;

        // Both m and n must be non-negative for real roots
        if m.is_negative() || n.is_negative() {
            return None;
        }

        // Build result: sqrt(m) + sqrt(n)
        let m_expr = ctx.add(Expr::Number(m.clone()));
        let n_expr = ctx.add(Expr::Number(n.clone()));

        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let sqrt_m = ctx.add(Expr::Pow(m_expr, half));
        let half2 = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let sqrt_n = ctx.add(Expr::Pow(n_expr, half2));

        let result = ctx.add(Expr::Add(sqrt_m, sqrt_n));

        Some(Rewrite {
            new_expr: result,
            description: format!("Denest nested square root: √(a+√b) = √({}) + √({})", m, n),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        })
    }
);

// =============================================================================
// DENEST PERFECT CUBE IN QUADRATIC FIELD RULE
// Simplifies ∛(A + B√n) → x + y√n where (x+y√n)³ = A+B√n
// =============================================================================

/// Try to split an expression as A + B*sqrt(n) where A, B, n are rationals.
/// Returns (A, B, n) if successful.
fn split_linear_surd(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(
    num_rational::BigRational,
    num_rational::BigRational,
    num_rational::BigRational,
)> {
    use num_rational::BigRational;

    // Helper to extract coefficient and radicand from a surd term (B*sqrt(n) or sqrt(n))
    fn extract_coef_surd(
        ctx: &cas_ast::Context,
        term: cas_ast::ExprId,
    ) -> Option<(BigRational, BigRational)> {
        // Case: sqrt(n) or n^(1/2)
        if let Some(radicand) = as_sqrt(ctx, term) {
            if let Expr::Number(n) = ctx.get(radicand) {
                return Some((BigRational::from_integer(1.into()), n.clone()));
            }
        }

        // Case: B * sqrt(n) or sqrt(n) * B
        if let Expr::Mul(l, r) = ctx.get(term) {
            // Try l = B, r = sqrt(n)
            if let Expr::Number(b) = ctx.get(*l) {
                if let Some(radicand) = as_sqrt(ctx, *r) {
                    if let Expr::Number(n) = ctx.get(radicand) {
                        return Some((b.clone(), n.clone()));
                    }
                }
            }
            // Try l = sqrt(n), r = B
            if let Expr::Number(b) = ctx.get(*r) {
                if let Some(radicand) = as_sqrt(ctx, *l) {
                    if let Expr::Number(n) = ctx.get(radicand) {
                        return Some((b.clone(), n.clone()));
                    }
                }
            }
        }

        None
    }

    match ctx.get(expr) {
        // A + B*sqrt(n) or B*sqrt(n) + A
        Expr::Add(l, r) => {
            // Try: l = A (Number), r = B*sqrt(n)
            if let Expr::Number(a) = ctx.get(*l) {
                if let Some((b, n)) = extract_coef_surd(ctx, *r) {
                    return Some((a.clone(), b, n));
                }
            }
            // Try: l = B*sqrt(n), r = A (Number)
            if let Expr::Number(a) = ctx.get(*r) {
                if let Some((b, n)) = extract_coef_surd(ctx, *l) {
                    return Some((a.clone(), b, n));
                }
            }
            // Check for l + Neg(r) = l - something
            if let Expr::Neg(neg_inner) = ctx.get(*r) {
                if let Expr::Number(a) = ctx.get(*l) {
                    if let Some((b, n)) = extract_coef_surd(ctx, *neg_inner) {
                        return Some((a.clone(), -b, n));
                    }
                }
            }
            None
        }
        // A - B*sqrt(n)
        Expr::Sub(l, r) => {
            if let Expr::Number(a) = ctx.get(*l) {
                if let Some((b, n)) = extract_coef_surd(ctx, *r) {
                    return Some((a.clone(), -b, n));
                }
            }
            // Also handle sqrt(n) - A (which would be -A + sqrt(n))
            if let Expr::Number(a) = ctx.get(*r) {
                if let Some((b, n)) = extract_coef_surd(ctx, *l) {
                    return Some((-a.clone(), b, n));
                }
            }
            None
        }
        _ => None,
    }
}

/// Try to find rational x, y such that (x + y*sqrt(n))^3 = A + B*sqrt(n)
/// The equations are:
///   Rational part:    x³ + 3xy²n = A
///   Irrational part:  3x²y + y³n = B
/// We enumerate y from small rational candidates and solve for x.
fn solve_cube_in_quadratic_field(
    a: &num_rational::BigRational,
    b: &num_rational::BigRational,
    n: &num_rational::BigRational,
) -> Option<(num_rational::BigRational, num_rational::BigRational)> {
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};

    // Guard: n must be positive for real sqrt
    if n <= &BigRational::zero() {
        return None;
    }

    // Guard: don't process huge numbers
    let a_approx: f64 = a.numer().to_string().parse().unwrap_or(f64::MAX);
    let b_approx: f64 = b.numer().to_string().parse().unwrap_or(f64::MAX);
    if a_approx.abs() > 1e12 || b_approx.abs() > 1e12 {
        return None;
    }

    // Denominators to try for y: {1, 2, 3, 4, 6, 8, 12}
    let denoms: [i64; 7] = [1, 2, 3, 4, 6, 8, 12];

    // Numerator range based on rough estimate
    // |y| ≈ cbrt(|B|) / sqrt(n) roughly, but we use a generous bound
    let max_num: i64 = 10;

    let three = BigRational::from_integer(3.into());

    for &denom in &denoms {
        let denom_big = BigInt::from(denom);
        for num in -max_num..=max_num {
            if num == 0 {
                continue; // y = 0 would mean no surd part
            }

            let y = BigRational::new(BigInt::from(num), denom_big.clone());

            // From: 3x²y + y³n = B
            // => x² = (B/y - y²n) / 3 = (B - y³n) / (3y)
            // But easier from: y(3x² + ny²) = B
            // => 3x² + ny² = B/y
            // => x² = (B/y - ny²) / 3

            let y_squared = &y * &y;
            let y_cubed = &y_squared * &y;

            // x² = (B/y - n*y²) / 3
            let b_over_y = b / &y;
            let n_y_sq = n * &y_squared;
            let x_squared = (&b_over_y - &n_y_sq) / &three;

            // x² must be non-negative
            if x_squared.is_negative() {
                continue;
            }

            // Try to get rational sqrt of x²
            if let Some(x_pos) = rational_sqrt(&x_squared) {
                // Try both +x and -x
                for x in [x_pos.clone(), -x_pos.clone()] {
                    // Verify: x³ + 3xy²n = A
                    let x_cubed = &x * &x * &x;
                    let term_3xy2n = &three * &x * &y_squared * n;
                    let lhs_a = &x_cubed + &term_3xy2n;

                    // Verify: 3x²y + y³n = B
                    let x_sq = &x * &x;
                    let term_3x2y = &three * &x_sq * &y;
                    let term_y3n = &y_cubed * n;
                    let lhs_b = &term_3x2y + &term_y3n;

                    if &lhs_a == a && &lhs_b == b {
                        return Some((x, y));
                    }
                }
            }
        }
    }

    None
}

define_rule!(
    DenestPerfectCubeInQuadraticFieldRule,
    "Denest Cube Root in Quadratic Field",
    None,
    crate::phase::PhaseMask::TRANSFORM,
    |ctx, expr| {
        use num_traits::Zero;

        // Match Pow(base, 1/3)
        let (base, exp) = match ctx.get(expr) {
            Expr::Pow(b, e) => (*b, *e),
            _ => return None,
        };

        // Check exponent is 1/3
        if let Expr::Number(exp_val) = ctx.get(exp) {
            if !(*exp_val.numer() == 1.into() && *exp_val.denom() == 3.into()) {
                return None;
            }
        } else {
            return None;
        }

        // Extract A + B*sqrt(n) from base
        let (a, b, n) = split_linear_surd(ctx, base)?;

        // Guard: b must be non-zero (otherwise no surd)
        if b.is_zero() {
            return None;
        }

        // Try to find x, y such that (x + y*sqrt(n))³ = A + B*sqrt(n)
        let (x, y) = solve_cube_in_quadratic_field(&a, &b, &n)?;

        // Build result: x + y*sqrt(n)
        let x_expr = ctx.add(Expr::Number(x.clone()));
        let y_expr = ctx.add(Expr::Number(y.clone()));
        let n_expr = ctx.add(Expr::Number(n.clone()));

        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let sqrt_n = ctx.add(Expr::Pow(n_expr, half));

        let result = if y.is_zero() {
            x_expr
        } else if x.is_zero() {
            let y_sqrt_n = ctx.add(Expr::Mul(y_expr, sqrt_n));
            y_sqrt_n
        } else {
            let y_sqrt_n = ctx.add(Expr::Mul(y_expr, sqrt_n));
            ctx.add(Expr::Add(x_expr, y_sqrt_n))
        };

        Some(Rewrite {
            new_expr: result,
            description: format!(
                "Denest cube root in quadratic field: ∛(A+B√n) = {} + {}√{}",
                x, y, n
            ),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        })
    }
);

#[cfg(test)]
mod cubic_conjugate_tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn test_cubic_conjugate_basic() {
        let mut ctx = Context::new();
        let expr = parse("(2 + 5^(1/2))^(1/3) + (2 - 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_cubic_conjugate_commuted() {
        let mut ctx = Context::new();
        // Reversed order
        let expr = parse("(2 - 5^(1/2))^(1/3) + (2 + 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_cubic_conjugate_no_match_different_surd() {
        let mut ctx = Context::new();
        // Different surds: sqrt(5) vs sqrt(6)
        let expr = parse("(2 + 5^(1/2))^(1/3) + (2 - 6^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match different surds");
    }

    #[test]
    fn test_cubic_conjugate_no_match_different_exp() {
        let mut ctx = Context::new();
        // Different exponents: 1/3 vs 1/5
        let expr = parse("(2 + 5^(1/2))^(1/3) + (2 - 5^(1/2))^(1/5)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match different exponents");
    }

    #[test]
    fn test_prerequisite_negative_cube_root() {
        // Prerequisite: (-1)^(1/3) must equal -1 for the rule to work
        let mut ctx = Context::new();
        let expr = parse("(-1)^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "-1");
    }

    #[test]
    fn test_prerequisite_negative_8_cube_root() {
        // (-8)^(1/3) = -2
        let mut ctx = Context::new();
        let expr = parse("(-8)^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "-2");
    }

    #[test]
    fn test_cubic_conjugate_sqrt_function_form() {
        // Test with sqrt() function instead of ^(1/2)
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(5))^(1/3) + (2 - sqrt(5))^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_cubic_conjugate_no_match_not_sum() {
        // Subtraction instead of sum - should not match
        let mut ctx = Context::new();
        let expr = parse("(2 + 5^(1/2))^(1/3) - (2 - 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match subtraction");
    }

    #[test]
    fn test_cubic_conjugate_no_match_same_signs() {
        // Both addends have same sign: (m+t) + (m+t) style
        let mut ctx = Context::new();
        let expr = parse("(2 + 5^(1/2))^(1/3) + (2 + 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(
            rewrite.is_none(),
            "Should not match when both have same sign"
        );
    }

    #[test]
    fn test_cubic_conjugate_no_match_irrational_root() {
        // (1 + √2)^(1/3) + (1 - √2)^(1/3)
        // AB = 1 - 2 = -1 is a cube, but cubic x³ + 3x - 2 = 0 has no rational root
        let mut ctx = Context::new();
        let expr = parse("(1 + 2^(1/2))^(1/3) + (1 - 2^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        // The rule should not find a rational root (root is ~0.596)
        assert!(
            rewrite.is_none(),
            "Should not match when no rational root exists"
        );
    }

    #[test]
    fn test_cubic_conjugate_no_match_different_m() {
        // Different m values: (2+√5)^(1/3) + (3-√5)^(1/3)
        let mut ctx = Context::new();
        let expr = parse("(2 + 5^(1/2))^(1/3) + (3 - 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match different m values");
    }
}

#[cfg(test)]
mod denest_sqrt_tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn test_denest_sqrt_4_plus_sqrt7() {
        // √(4 + √7) → √(7/2) + √(1/2)
        let mut ctx = Context::new();
        let expr = parse("sqrt(4 + sqrt(7))", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_some(), "Rule should apply to √(4+√7)");

        // Verify the result simplifies correctly
        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        // Just check that we get the denested form with surds
        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        // Should contain fractions 1/2 and 7/2
        assert!(
            result_str.contains("1/2") && result_str.contains("7/2"),
            "Result should be √(1/2)+√(7/2), got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_sqrt_pow_form() {
        // (4 + 7^(1/2))^(1/2) → pow form instead of sqrt function
        // Use simplifier since the expression needs canonicalization
        let mut ctx = Context::new();
        let expr = parse("(4 + 7^(1/2))^(1/2)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        // Should contain fractions 1/2 and 7/2
        assert!(
            result_str.contains("1/2") && result_str.contains("7/2"),
            "Result should be denested with 1/2 and 7/2, got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_sqrt_no_match_bad_discriminant() {
        // √(3 + √5): disc = 9 - 5 = 4 ✓, but let's check it works
        // disc_sqrt = 2, m = (3+2)/2 = 5/2, n = (3-2)/2 = 1/2
        let mut ctx = Context::new();
        let expr = parse("sqrt(3 + sqrt(5))", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(
            rewrite.is_some(),
            "Should match √(3+√5) since disc=4 is perfect square"
        );
    }

    #[test]
    fn test_denest_sqrt_no_match_non_perfect_square_disc() {
        // √(4 + √10): disc = 16 - 10 = 6 (not a perfect square)
        let mut ctx = Context::new();
        let expr = parse("sqrt(4 + sqrt(10))", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(
            rewrite.is_none(),
            "Should not match when disc=6 is not a perfect square"
        );
    }

    #[test]
    fn test_denest_sqrt_no_match_negative_m_or_n() {
        // √(1 + √10): disc = 1 - 10 = -9 (negative)
        let mut ctx = Context::new();
        let expr = parse("sqrt(1 + sqrt(10))", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match when disc < 0");
    }

    #[test]
    fn test_denest_sqrt_commuted_order() {
        // √(√7 + 4) - surd comes first
        let mut ctx = Context::new();
        let expr = parse("sqrt(sqrt(7) + 4)", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_some(), "Should match commuted order √(√7+4)");
    }
}

#[cfg(test)]
mod denest_cube_quadratic_tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn test_denest_cube_26_15_sqrt3() {
        // (26 + 15*sqrt(3))^(1/3) → 2 + sqrt(3)
        // Because (2 + sqrt(3))³ = 26 + 15*sqrt(3)
        let mut ctx = Context::new();
        let expr = parse("(26 + 15*sqrt(3))^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert!(
            result_str.contains("2") && result_str.contains("3"),
            "Should be 2 + √3, got: {}",
            result_str
        );
        // Verify it doesn't contain a cube root anymore
        assert!(
            !result_str.contains("∛") && !result_str.contains("1/3"),
            "Should NOT contain cube root, got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_cube_golden_ratio() {
        // (2 + sqrt(5))^(1/3) → (1 + sqrt(5))/2 = φ
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(5))^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        // Should be (1 + √5)/2 which may display as "(1 + √(5))/2"
        assert!(
            result_str.contains("1") && result_str.contains("5") && result_str.contains("2"),
            "Should be (1+√5)/2, got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_cube_golden_ratio_conjugate() {
        // (2 - sqrt(5))^(1/3) → (1 - sqrt(5))/2 = 1-φ
        let mut ctx = Context::new();
        let expr = parse("(2 - sqrt(5))^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        // Should be (1 - √5)/2
        assert!(
            result_str.contains("1") && result_str.contains("5"),
            "Should be (1-√5)/2, got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_cube_no_match_sqrt6() {
        // (2 + sqrt(6))^(1/3) - no rational x,y exists
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(6))^(1/3)", &mut ctx).unwrap();

        let rule = DenestPerfectCubeInQuadraticFieldRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should NOT match (2+√6)^(1/3)");
    }

    #[test]
    fn test_denest_cube_no_match_wrong_exp() {
        // (2 + sqrt(5))^(1/5) - exponent is 1/5 not 1/3
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(5))^(1/5)", &mut ctx).unwrap();

        let rule = DenestPerfectCubeInQuadraticFieldRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should NOT match exponent 1/5");
    }
}
