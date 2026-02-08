use crate::define_rule;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{BuiltinFn, Context, Expr};
use num_traits::{Signed, Zero};

define_rule!(RootDenestingRule, "Root Denesting", |ctx, expr| {
    // Extract shape of expr: Function(sqrt, args) or Pow(b, 1/2)
    enum RootShape {
        Sqrt(cas_ast::ExprId),
        HalfPow(cas_ast::ExprId),
        Other,
    }
    let shape = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            RootShape::Sqrt(args[0])
        }
        Expr::Pow(b, e) => {
            let b = *b;
            let e = *e;
            if let Expr::Number(n) = ctx.get(e) {
                if *n.numer() == 1.into() && *n.denom() == 2.into() {
                    RootShape::HalfPow(b)
                } else {
                    RootShape::Other
                }
            } else {
                RootShape::Other
            }
        }
        _ => RootShape::Other,
    };

    let inner = match shape {
        RootShape::Sqrt(i) | RootShape::HalfPow(i) => Some(i),
        RootShape::Other => None,
    };

    let inner = inner?;

    let (a, b, is_add) = match ctx.get(inner) {
        Expr::Add(l, r) => (*l, *r, true),
        Expr::Sub(l, r) => (*l, *r, false),
        _ => return None,
    };

    // Helper to identify if a term is C*sqrt(D) or sqrt(D)
    // Returns (Option<C>, D). If C is None, it means 1.
    fn analyze_sqrt_term(
        ctx: &Context,
        e: cas_ast::ExprId,
    ) -> Option<(Option<cas_ast::ExprId>, cas_ast::ExprId)> {
        match ctx.get(e) {
            Expr::Function(fname, fargs)
                if ctx.is_builtin(*fname, BuiltinFn::Sqrt) && fargs.len() == 1 =>
            {
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
                        Expr::Function(fname, fargs)
                            if ctx.is_builtin(*fname, BuiltinFn::Sqrt) && fargs.len() == 1 =>
                        {
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
                // Clone all values upfront to release the immutable borrow
                let val_a = val_a.clone();
                let val_c = val_c.clone();
                let val_d = val_d.clone();

                let val_c2 = &val_c * &val_c;
                let val_beff = &val_c2 * &val_d;
                let val_a2 = &val_a * &val_a;
                let val_delta = &val_a2 - &val_beff;

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
                        let term1 = ctx.call("sqrt", vec![term1_frac]);

                        let term2_num = ctx.add(Expr::Sub(term_a, z_val));
                        let term2_frac = ctx.add(Expr::Div(term2_num, two));
                        let term2 = ctx.call("sqrt", vec![term2_frac]);

                        // Check sign of C - use our cloned value
                        let c_is_negative = val_c < num_rational::BigRational::zero();

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

                        // Simple description - didactic layer will add detailed substeps
                        return Some(
                            crate::rule::Rewrite::new(new_expr)
                                .desc("Denest square root")
                                .local(expr, new_expr),
                        );
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
        let arg = if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 {
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

            let vars = cas_ast::collect_variables(ctx, arg);
            if vars.len() == 1 {
                let var = vars.iter().next()?;
                if let Ok(poly) = Polynomial::from_expr(ctx, arg, var) {
                    // First: Try to detect perfect square with rational coefficients
                    // For ax² + bx + c to be (dx + e)², we need:
                    // - a = d², c = e², b = 2de
                    // - Equivalently: b² = 4ac (discriminant = 0)
                    if poly.degree() == 2 && poly.coeffs.len() >= 3 {
                        let a = poly.coeffs.get(2).cloned();
                        let b = poly.coeffs.get(1).cloned();
                        let c = poly.coeffs.first().cloned();

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
                                            .unwrap_or_else(num_rational::BigRational::zero)
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
                                    let abs_linear = ctx.call("abs", vec![linear]);

                                    return Some(
                                        Rewrite::new(abs_linear)
                                            .desc("Simplify perfect square root"),
                                    );
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

                                let abs_base = ctx.call("abs", vec![base]);

                                let term1 = if k == 1 {
                                    abs_base
                                } else {
                                    let k_expr = ctx.num(k as i64);
                                    ctx.add(Expr::Pow(abs_base, k_expr))
                                };

                                if rem == 0 {
                                    return Some(
                                        Rewrite::new(term1).desc("Simplify perfect square root"),
                                    );
                                } else {
                                    let sqrt_base = ctx.call("sqrt", vec![base]);
                                    let new_expr = smart_mul(ctx, term1, sqrt_base);
                                    return Some(
                                        Rewrite::new(new_expr).desc("Simplify square root factors"),
                                    );
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
pub(crate) fn rational_sqrt(r: &num_rational::BigRational) -> Option<num_rational::BigRational> {
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
