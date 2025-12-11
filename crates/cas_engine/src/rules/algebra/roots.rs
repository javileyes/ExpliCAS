use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr};
use num_traits::Zero;

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

                if val_delta >= num_rational::BigRational::zero()
                    && val_delta.is_integer() {
                        let int_delta = val_delta.to_integer();
                        let sqrt_delta = int_delta.sqrt();

                        if sqrt_delta.clone() * sqrt_delta.clone() == int_delta {
                            // Perfect square!
                            let z_val = ctx.add(Expr::Number(
                                num_rational::BigRational::from_integer(sqrt_delta),
                            ));

                            // Found Z!
                            // Result = sqrt((A+Z)/2) +/- sqrt((A-Z)/2)
                            let two = ctx.num(2);

                            let term1_num = ctx.add(Expr::Add(term_a, z_val));
                            let term1_frac = ctx.add(Expr::Div(term1_num, two));
                            let term1 =
                                ctx.add(Expr::Function("sqrt".to_string(), vec![term1_frac]));

                            let term2_num = ctx.add(Expr::Sub(term_a, z_val));
                            let term2_frac = ctx.add(Expr::Div(term2_num, two));
                            let term2 =
                                ctx.add(Expr::Function("sqrt".to_string(), vec![term2_frac]));

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
                                    });
                                } else {
                                    let sqrt_base =
                                        ctx.add(Expr::Function("sqrt".to_string(), vec![base]));
                                    let new_expr = ctx.add(Expr::Mul(term1, sqrt_base));
                                    return Some(Rewrite {
                                        new_expr,
                                        description: "Simplify square root factors".to_string(),
                                        before_local: None,
                                        after_local: None,
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
