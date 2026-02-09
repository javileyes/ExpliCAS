//! Tan half-angle and trigonometric quotient rules.
//!
//! This module contains rules for:
//! - Hyperbolic half-angle identities: cosh²(x/2) = (cosh(x)+1)/2
//! - Generalized sin·cos contraction: k·sin(t)·cos(t) = (k/2)·sin(2t)
//! - Trig quotient simplification: sin/cos → tan
//! - Tan double angle contraction

use crate::define_rule;
use crate::helpers::as_div;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr, ExprId};
use num_traits::One;

// =============================================================================

define_rule!(
    HyperbolicHalfAngleSquaresRule,
    "Hyperbolic Half-Angle Squares",
    |ctx, expr| {
        if let Expr::Pow(base, exp) = ctx.get(expr) {
            // Check if exponent is 2
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n != num_rational::BigRational::from_integer(2.into()) {
                    return None;
                }
            } else {
                return None;
            }

            // Check if base is cosh(x/2) or sinh(x/2)
            let (fn_id, args) = if let Expr::Function(fn_id, args) = ctx.get(*base) {
                (*fn_id, args.clone())
            } else {
                return None;
            };
            {
                let builtin = ctx.builtin_of(fn_id);
                let is_cosh = matches!(builtin, Some(BuiltinFn::Cosh));
                let is_sinh = matches!(builtin, Some(BuiltinFn::Sinh));
                if (is_cosh || is_sinh) && args.len() == 1 {
                    let arg = args[0];

                    // Check if argument is x/2 or (1/2)*x
                    let full_angle = match ctx.get(arg) {
                        Expr::Div(num, den) => {
                            if let Expr::Number(d) = ctx.get(*den) {
                                if *d == num_rational::BigRational::from_integer(2.into()) {
                                    Some(*num)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        Expr::Mul(l, r) => {
                            let half = num_rational::BigRational::new(1.into(), 2.into());
                            if let Expr::Number(n) = ctx.get(*l) {
                                if *n == half {
                                    Some(*r)
                                } else {
                                    None
                                }
                            } else if let Expr::Number(n) = ctx.get(*r) {
                                if *n == half {
                                    Some(*l)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };

                    if let Some(x) = full_angle {
                        let cosh_x = ctx.call_builtin(cas_ast::BuiltinFn::Cosh, vec![x]);
                        let one = ctx.num(1);
                        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
                            1.into(),
                            2.into(),
                        )));

                        if is_cosh {
                            // cosh(x/2)² → (cosh(x)+1)/2
                            let sum = ctx.add(Expr::Add(cosh_x, one));
                            let result = ctx.add(Expr::Mul(half, sum));
                            return Some(Rewrite::new(result).desc("cosh²(x/2) = (cosh(x)+1)/2"));
                        } else {
                            // sinh(x/2)² → (cosh(x)-1)/2
                            let diff = ctx.add(Expr::Sub(cosh_x, one));
                            let result = ctx.add(Expr::Mul(half, diff));
                            return Some(Rewrite::new(result).desc("sinh²(x/2) = (cosh(x)-1)/2"));
                        }
                    }
                }
            }
        }
        None
    }
);

// =============================================================================
// GeneralizedSinCosContractionRule: k*sin(t)*cos(t) → (k/2)*sin(2t) for even k
// =============================================================================
// Extends DoubleAngleContractionRule to handle k*sin*cos where k is even (4, 6, 8, etc.)

define_rule!(
    GeneralizedSinCosContractionRule,
    "Generalized Sin Cos Contraction",
    |ctx, expr| {
        // Only match Mul at top level
        if let Expr::Mul(_l, _r) = ctx.get(expr) {
            // Flatten the multiplication to find all factors
            let factors = crate::nary::mul_leaves(ctx, expr);

            if factors.len() < 3 {
                return None;
            }

            // Look for: coefficient (even ≥ 4), sin(t), cos(t)
            let mut coef_idx: Option<usize> = None;
            let mut coef_val: Option<num_rational::BigRational> = None;
            let mut sin_idx: Option<usize> = None;
            let mut sin_arg: Option<ExprId> = None;
            let mut cos_idx: Option<usize> = None;
            let mut cos_arg: Option<ExprId> = None;

            for (i, &factor) in factors.iter().enumerate() {
                // Check for numeric coefficient
                if let Expr::Number(n) = ctx.get(factor) {
                    // Check if n is even and ≥ 4
                    let two = num_rational::BigRational::from_integer(2.into());
                    let four = num_rational::BigRational::from_integer(4.into());
                    if n >= &four && (n / &two).is_integer() {
                        coef_idx = Some(i);
                        coef_val = Some(n.clone());
                        continue;
                    }
                }
                // Check for sin(t)
                if let Expr::Function(fn_id, args) = ctx.get(factor) {
                    let builtin = ctx.builtin_of(*fn_id);
                    if matches!(builtin, Some(BuiltinFn::Sin))
                        && args.len() == 1
                        && sin_idx.is_none()
                    {
                        sin_idx = Some(i);
                        sin_arg = Some(args[0]);
                        continue;
                    }
                    if matches!(builtin, Some(BuiltinFn::Cos))
                        && args.len() == 1
                        && cos_idx.is_none()
                    {
                        cos_idx = Some(i);
                        cos_arg = Some(args[0]);
                        continue;
                    }
                }
            }

            // If we found all three and sin_arg == cos_arg (same angle)
            if let (Some(c_i), Some(c_val), Some(s_i), Some(s_arg), Some(o_i), Some(c_arg)) =
                (coef_idx, coef_val, sin_idx, sin_arg, cos_idx, cos_arg)
            {
                // Check that sin and cos have the same argument
                if s_arg == c_arg {
                    // Build (k/2)*sin(2*t)
                    let two = num_rational::BigRational::from_integer(2.into());
                    let half_coef = c_val / &two;
                    let half_coef_expr = ctx.add(Expr::Number(half_coef));

                    let two_expr = ctx.num(2);
                    let double_arg = ctx.add(Expr::Mul(two_expr, s_arg));
                    let sin_2t = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![double_arg]);

                    // Build the result: (k/2)*sin(2t) * [remaining factors]
                    let contracted = ctx.add(Expr::Mul(half_coef_expr, sin_2t));

                    // Collect remaining factors
                    let mut remaining: Vec<ExprId> = Vec::new();
                    for (j, &f) in factors.iter().enumerate() {
                        if j != c_i && j != s_i && j != o_i {
                            remaining.push(f);
                        }
                    }

                    let result = if remaining.is_empty() {
                        contracted
                    } else {
                        let mut acc = contracted;
                        for &f in remaining.iter() {
                            acc = ctx.add(Expr::Mul(acc, f));
                        }
                        acc
                    };

                    return Some(Rewrite::new(result).desc("k·sin(t)·cos(t) = (k/2)·sin(2t)"));
                }
            }
        }
        None
    }
);

// =============================================================================
// TrigQuotientToNamedRule: sin(t)/cos(t) → tan(t), 1/cos(t) → sec(t), etc.
// =============================================================================
// Canonicalize trig quotients to named functions for better normalization.
// This ensures that `sin(u)/cos(u)` and `tan(u)` converge to the same form.

define_rule!(
    TrigQuotientToNamedRule,
    "Trig Quotient to Named Function",
    |ctx, expr| {
        let (num, den) = as_div(ctx, expr)?;
        // Pattern: 1/cos(t) → sec(t), 1/sin(t) → csc(t)
        if let Expr::Number(n) = ctx.get(num) {
            if n.is_one() {
                let (fn_id, args) = if let Expr::Function(fn_id, args) = ctx.get(den) {
                    (*fn_id, args.clone())
                } else {
                    return None;
                };
                {
                    if args.len() == 1 {
                        let arg = args[0];
                        let result_info = match ctx.builtin_of(fn_id) {
                            Some(BuiltinFn::Cos) => Some(("sec", "cos")),
                            Some(BuiltinFn::Sin) => Some(("csc", "sin")),
                            _ => None,
                        };
                        if let Some((rn, orig_name)) = result_info {
                            let result = ctx.call(rn, vec![arg]);
                            return Some(
                                Rewrite::new(result)
                                    .desc_lazy(|| format!("1/{}(t) = {}(t)", orig_name, rn)),
                            );
                        }
                    }
                }
            }
        }

        // Pattern: sin(t)/cos(t) → tan(t), cos(t)/sin(t) → cot(t)
        let matched = match (ctx.get(num), ctx.get(den)) {
            (Expr::Function(num_fn_id, num_args), Expr::Function(den_fn_id, den_args)) => {
                Some((*num_fn_id, num_args.clone(), *den_fn_id, den_args.clone()))
            }
            _ => None,
        };
        if let Some((num_fn_id, num_args, den_fn_id, den_args)) = matched {
            if num_args.len() == 1 && den_args.len() == 1 {
                let num_arg = num_args[0];
                let den_arg = den_args[0];

                let num_builtin = ctx.builtin_of(num_fn_id);
                let den_builtin = ctx.builtin_of(den_fn_id);

                // Check same argument
                if crate::ordering::compare_expr(ctx, num_arg, den_arg) == std::cmp::Ordering::Equal
                {
                    let result_name = match (num_builtin, den_builtin) {
                        (Some(BuiltinFn::Sin), Some(BuiltinFn::Cos)) => Some(("tan", "sin", "cos")),
                        (Some(BuiltinFn::Cos), Some(BuiltinFn::Sin)) => Some(("cot", "cos", "sin")),
                        _ => None,
                    };
                    if let Some((rn, num_display, den_display)) = result_name {
                        let result = ctx.call(rn, vec![num_arg]);
                        return Some(Rewrite::new(result).desc_lazy(|| {
                            format!("{}/{}(t) = {}(t)", num_display, den_display, rn)
                        }));
                    }
                }
            }
        }
        None
    }
);

// =============================================================================
// TanDoubleAngleContractionRule: 2*tan(t)/(1 - tan(t)²) → tan(2*t)
// =============================================================================
// This contracts the expanded tan(2t) form back to the double angle form.
// Prevents the engine from creating deeply nested fractions when tan²(t)
// appears in denominators.

define_rule!(
    TanDoubleAngleContractionRule,
    "Tan Double Angle Contraction",
    |ctx, expr| {
        // Match Div(numerator, denominator)
        let (num, den) = if let Expr::Div(num, den) = ctx.get(expr) {
            (*num, *den)
        } else {
            return None;
        };
        {
            // Numerator should be 2*tan(t) (or tan(t)*2)
            let tan_arg = if let Expr::Mul(l, r) = ctx.get(num) {
                let (l, r) = (*l, *r);
                let (coeff, tan_part) = if let Expr::Number(n) = ctx.get(l) {
                    if *n == num_rational::BigRational::from_integer(2.into()) {
                        (true, r)
                    } else {
                        (false, l) // dummy
                    }
                } else if let Expr::Number(n) = ctx.get(r) {
                    if *n == num_rational::BigRational::from_integer(2.into()) {
                        (true, l)
                    } else {
                        (false, l) // dummy
                    }
                } else {
                    (false, l) // dummy
                };

                if coeff {
                    if let Expr::Function(fn_id, args) = ctx.get(tan_part) {
                        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tan)) && args.len() == 1
                        {
                            Some(args[0])
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(t) = tan_arg {
                // Denominator should be 1 - tan(t)² (or equivalently: 1 + (-tan(t)²) or Sub(1, tan(t)²))
                let den_matches = if let Expr::Sub(one_part, tan2_part) = ctx.get(den) {
                    let (one_part, tan2_part) = (*one_part, *tan2_part);
                    // Check 1 - tan(t)²
                    let one_ok = matches!(ctx.get(one_part), Expr::Number(n) if n.is_one());
                    let tan2_ok = if let Expr::Pow(base, exp) = ctx.get(tan2_part) {
                        let exp_is_2 = matches!(ctx.get(*exp), Expr::Number(n)
                            if *n == num_rational::BigRational::from_integer(2.into()));
                        if exp_is_2 {
                            if let Expr::Function(fn_id, args) = ctx.get(*base) {
                                matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tan))
                                    && args.len() == 1
                                    && crate::ordering::compare_expr(ctx, args[0], t)
                                        == std::cmp::Ordering::Equal
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    one_ok && tan2_ok
                } else if let Expr::Add(l, r) = ctx.get(den) {
                    let (l, r) = (*l, *r);
                    // Check 1 + (-tan(t)²) i.e. 1 + Neg(...)
                    let (one_part, neg_part) = if matches!(ctx.get(l), Expr::Number(n) if n.is_one())
                    {
                        (l, r)
                    } else if matches!(ctx.get(r), Expr::Number(n) if n.is_one()) {
                        (r, l)
                    } else {
                        return None;
                    };
                    let _ = one_part;

                    if let Expr::Neg(inner) = ctx.get(neg_part) {
                        if let Expr::Pow(base, exp) = ctx.get(*inner) {
                            let exp_is_2 = matches!(ctx.get(*exp), Expr::Number(n)
                                if *n == num_rational::BigRational::from_integer(2.into()));
                            if exp_is_2 {
                                if let Expr::Function(fn_id, args) = ctx.get(*base) {
                                    matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tan))
                                        && args.len() == 1
                                        && crate::ordering::compare_expr(ctx, args[0], t)
                                            == std::cmp::Ordering::Equal
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                if den_matches {
                    // Build tan(2*t)
                    let two = ctx.num(2);
                    let double_t = ctx.add(Expr::Mul(two, t));
                    let result = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![double_t]);
                    return Some(Rewrite::new(result).desc("2·tan(t)/(1-tan²(t)) = tan(2t)"));
                }
            }
        }
        None
    }
);
