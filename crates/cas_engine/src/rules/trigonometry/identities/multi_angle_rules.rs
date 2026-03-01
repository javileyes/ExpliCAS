//! Multi-angle expansion rules (triple, quintuple, recursive) and trig-quotient helpers.
//!
//! Extracted from `expansion_rules.rs` to keep module size manageable.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::expr_destructure::{as_mul, as_pow};
use cas_math::expr_rewrite::smart_mul;
use cas_math::trig_multi_angle_support::{
    is_inside_trig_sum_quotient_with_ancestors, is_trivial_angle, try_rewrite_quintuple_angle_expr,
    try_rewrite_triple_angle_expr,
};
use num_traits::{One, Zero};

// Triple Angle Shortcut Rule: sin(3x) → 3sin(x) - 4sin³(x), cos(3x) → 4cos³(x) - 3cos(x)
// This is a performance optimization to avoid recursive expansion via double-angle rules.
// Reduces ~23 rewrites to ~3-5 for triple angle expressions.
define_rule!(
    TripleAngleRule,
    "Triple Angle Identity",
    |ctx, expr, parent_ctx| {
        // GUARD 1: Skip if marked for protection
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sum_quotient_protected(expr) {
                return None;
            }
        }

        // GUARD 2: Skip if inside sum-quotient pattern (defer to SinCosSumQuotientRule)
        if is_inside_trig_sum_quotient_with_ancestors(ctx, parent_ctx.all_ancestors()) {
            return None;
        }

        let rewrite = try_rewrite_triple_angle_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Quintuple Angle Rule: sin(5x) → 16sin⁵(x) - 20sin³(x) + 5sin(x)
// This is a direct expansion to avoid recursive explosion via double/triple angle.
define_rule!(
    QuintupleAngleRule,
    "Quintuple Angle Identity",
    |ctx, expr, parent_ctx| {
        // GUARD 1: Skip if marked for protection
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sum_quotient_protected(expr) {
                return None;
            }
        }

        // GUARD 2: Skip if inside sum-quotient pattern
        if is_inside_trig_sum_quotient_with_ancestors(ctx, parent_ctx.all_ancestors()) {
            return None;
        }

        let rewrite = try_rewrite_quintuple_angle_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    RecursiveTrigExpansionRule,
    "Recursive Trig Expansion",
    |ctx, expr, parent_ctx| {
        // GUARD 1: Skip if this trig function is marked for protection
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sum_quotient_protected(expr) {
                return None;
            }
            // GUARD 1b: Skip if sin(4x) identity pattern detected
            // This prevents sin(4*t) from expanding before Sin4xIdentityZeroRule can fire
            if marks.has_sin4x_identity_pattern {
                return None;
            }
        }

        // GUARD 2: Skip if we're inside a potential sum-quotient pattern
        // This heuristic checks: if the trig function is inside a Div, and both
        // numerator and denominator are Add/Sub of trig functions, defer to
        // SinCosSumQuotientRule instead of expanding.
        if is_inside_trig_sum_quotient_with_ancestors(ctx, parent_ctx.all_ancestors()) {
            return None;
        }

        let (fn_id, args) = match ctx.get(expr) {
            Expr::Function(fn_id, args) => (*fn_id, args.clone()),
            _ => return None,
        };
        {
            let builtin = ctx.builtin_of(fn_id);
            let is_sin = matches!(builtin, Some(BuiltinFn::Sin));
            let is_cos = matches!(builtin, Some(BuiltinFn::Cos));
            if args.len() == 1 && (is_sin || is_cos) {
                // Check for n * x where n is integer > 2
                let inner = args[0];

                let (n_val, x_val) = if let Some((l, r)) = as_mul(ctx, inner) {
                    if let Expr::Number(n) = ctx.get(l) {
                        if n.is_integer() {
                            (n.to_integer(), r)
                        } else {
                            return None;
                        }
                    } else if let Expr::Number(n) = ctx.get(r) {
                        if n.is_integer() {
                            (n.to_integer(), l)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                };

                if n_val > num_bigint::BigInt::from(2) && n_val <= num_bigint::BigInt::from(6) {
                    // GUARD: Only expand sin(n*x) for small n (3-6).
                    // For n > 6, the expansion grows exponentially without benefit.
                    // This prevents catastrophic expansion like sin(671*x) → 670 recursive steps.

                    // Rewrite sin(nx) -> sin((n-1)x + x)

                    let n_minus_1 = n_val.clone() - 1;
                    let n_minus_1_expr = ctx.add(Expr::Number(
                        num_rational::BigRational::from_integer(n_minus_1),
                    ));
                    let term_nm1 = smart_mul(ctx, n_minus_1_expr, x_val);

                    // sin(nx) = sin((n-1)x)cos(x) + cos((n-1)x)sin(x)
                    // cos(nx) = cos((n-1)x)cos(x) - sin((n-1)x)sin(x)

                    let sin_nm1 = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![term_nm1]);
                    let cos_nm1 = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![term_nm1]);
                    let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![x_val]);
                    let cos_x = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![x_val]);

                    if is_sin {
                        let t1 = smart_mul(ctx, sin_nm1, cos_x);
                        let t2 = smart_mul(ctx, cos_nm1, sin_x);
                        let new_expr = ctx.add(Expr::Add(t1, t2));
                        return Some(
                            Rewrite::new(new_expr)
                                .desc_lazy(|| format!("sin({}x) expansion", n_val)),
                        );
                    } else {
                        // cos
                        let t1 = smart_mul(ctx, cos_nm1, cos_x);
                        let t2 = smart_mul(ctx, sin_nm1, sin_x);
                        let new_expr = ctx.add(Expr::Sub(t1, t2));
                        return Some(
                            Rewrite::new(new_expr)
                                .desc_lazy(|| format!("cos({}x) expansion", n_val)),
                        );
                    }
                }
            }
        }
        None
    }
);

define_rule!(
    CanonicalizeTrigSquareRule,
    "Canonicalize Trig Square",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        // cos^n(x) -> (1 - sin^2(x))^(n/2) for even n
        if let Some((base, exp)) = as_pow(ctx, expr) {
            let n_opt = if let Expr::Number(n) = ctx.get(exp) {
                Some(n.clone())
            } else {
                None
            };

            if let Some(n) = n_opt {
                if n.is_integer()
                    && n.to_integer() % 2 == 0.into()
                    && n > num_rational::BigRational::zero()
                {
                    // Limit power to avoid explosion? Let's say <= 4 for now.
                    if n <= num_rational::BigRational::from_integer(4.into()) {
                        if let Expr::Function(fn_id, args) = ctx.get(base) {
                            if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) && args.len() == 1 {
                                let arg = args[0];
                                // (1 - sin^2(x))^(n/2)
                                let one = ctx.num(1);
                                let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![arg]);
                                let two = ctx.num(2);
                                let sin_sq = ctx.add(Expr::Pow(sin_x, two));
                                let base_term = ctx.add(Expr::Sub(one, sin_sq));

                                let half_n = n / num_rational::BigRational::from_integer(2.into());

                                if half_n.is_one() {
                                    return Some(Rewrite::new(base_term).desc("cos^2(x) -> 1 - sin^2(x)"));
                                } else {
                                    let half_n_expr = ctx.add(Expr::Number(half_n));
                                    let new_expr = ctx.add(Expr::Pow(base_term, half_n_expr));
                                    return Some(Rewrite::new(new_expr).desc("cos^2k(x) -> (1 - sin^2(x))^k"));
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

// =============================================================================
// Triple Angle CONTRACTION Rule
// =============================================================================
//
// Contracts the expanded form back into a triple angle:
//   3·sin(θ) − 4·sin³(θ)  →  sin(3θ)
//   4·cos³(θ) − 3·cos(θ)  →  cos(3θ)
//
// This is the reverse of TripleAngleRule. It fires on the additive form (Sub/Add
// nodes) and looks for matching pairs of linear and cubic trig terms.
//
// Cycle safety: the contraction produces sin(3θ) or cos(3θ). If the argument
// θ is compound (e.g. u²+1), distribution rewrites 3θ → 3u²+3, which no
// longer matches the Mul(3,x) pattern required by TripleAngleRule, so no
// reverse expansion occurs → no cycle.

define_rule!(
    TripleAngleContractionRule,
    "Triple Angle Contraction",
    |ctx, expr| {
        // Use add_terms_signed to properly decompose Sub(a,b) → [(a,+), (b,−)]
        let signed_terms = crate::nary::add_terms_signed(ctx, expr);
        if signed_terms.len() < 2 {
            return None;
        }

        // For each term, try to decompose as:
        //   sign * coefficient * trig(arg)^power
        // where trig is sin or cos, power is 1 or 3.
        struct TrigTerm {
            index: usize,
            coeff: num_rational::BigRational, // total signed coefficient
            builtin: BuiltinFn,               // Sin or Cos
            arg: ExprId,                      // θ
            power: i64,                       // 1 or 3
        }

        fn decompose_trig_term(
            ctx: &cas_ast::Context,
            term: ExprId,
            sign: crate::nary::Sign,
        ) -> Option<(num_rational::BigRational, BuiltinFn, ExprId, i64)> {
            // The outer sign comes from add_terms_signed
            let outer_sign = num_rational::BigRational::from_integer(sign.to_i32().into());

            // Peel Neg: Neg(inner) → extra sign flip
            let (inner, neg_sign) = if let Expr::Neg(i) = ctx.get(term) {
                (*i, num_rational::BigRational::from_integer((-1).into()))
            } else {
                (term, num_rational::BigRational::from_integer(1.into()))
            };

            // Peel coefficient: Mul(k, rest) or k * rest
            let (coeff, core) = match ctx.get(inner) {
                Expr::Mul(l, r) => {
                    if let Expr::Number(n) = ctx.get(*l) {
                        (n.clone(), *r)
                    } else if let Expr::Number(n) = ctx.get(*r) {
                        (n.clone(), *l)
                    } else {
                        (num_rational::BigRational::from_integer(1.into()), inner)
                    }
                }
                _ => (num_rational::BigRational::from_integer(1.into()), inner),
            };
            let final_coeff = outer_sign * neg_sign * coeff;

            // Now core should be either trig(arg) or trig(arg)^3
            let (base, power) = if let Expr::Pow(b, e) = ctx.get(core) {
                if let Expr::Number(n) = ctx.get(*e) {
                    if n.is_integer() {
                        (*b, n.to_integer().try_into().ok().unwrap_or(0i64))
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            } else {
                (core, 1i64)
            };

            // base must be sin(arg) or cos(arg)
            if let Expr::Function(fn_id, args) = ctx.get(base) {
                if args.len() == 1 {
                    if let Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos)) = ctx.builtin_of(*fn_id) {
                        return Some((final_coeff, b, args[0], power));
                    }
                }
            }
            None
        }

        // Decompose all terms
        let mut trig_terms: Vec<TrigTerm> = Vec::new();
        for (i, (term, sign)) in signed_terms.iter().enumerate() {
            if let Some((c, b, a, p)) = decompose_trig_term(ctx, *term, *sign) {
                if p == 1 || p == 3 {
                    trig_terms.push(TrigTerm {
                        index: i,
                        coeff: c,
                        builtin: b,
                        arg: a,
                        power: p,
                    });
                }
            }
        }

        // Look for matching pairs: same builtin, same arg, powers 1 and 3
        for i in 0..trig_terms.len() {
            for j in 0..trig_terms.len() {
                if i == j {
                    continue;
                }
                let t1 = &trig_terms[i];
                let t3 = &trig_terms[j];

                // Must be same function and same argument
                if std::mem::discriminant(&t1.builtin) != std::mem::discriminant(&t3.builtin) {
                    continue;
                }
                if t1.power != 1 || t3.power != 3 {
                    continue;
                }
                if crate::ordering::compare_expr(ctx, t1.arg, t3.arg) != std::cmp::Ordering::Equal {
                    continue;
                }

                // Check coefficients
                let three = num_rational::BigRational::from_integer(3.into());
                let four = num_rational::BigRational::from_integer(4.into());

                let matched = match t1.builtin {
                    BuiltinFn::Sin => {
                        // Need: +3·sin(θ) − 4·sin³(θ) → sin(3θ)
                        // or k·(3·sin(θ) − 4·sin³(θ)) → k·sin(3θ)
                        // Check ratio: coeff1 * 4 == -coeff3 * 3
                        let lhs = &t1.coeff * &four;
                        let rhs = -(&t3.coeff) * &three;
                        lhs == rhs
                    }
                    BuiltinFn::Cos => {
                        // Need: 4·cos³(θ) − 3·cos(θ) → cos(3θ)
                        // Check ratio: coeff3 * 3 == -coeff1 * 4
                        let lhs = &t3.coeff * &three;
                        let rhs = -(&t1.coeff) * &four;
                        lhs == rhs
                    }
                    _ => false,
                };

                if !matched {
                    continue;
                }

                // Cycle guard: skip contraction when θ is trivial (Var,
                // Const, Mul(num,Var)) — TripleAngleRule handles those.
                // See is_trivial_angle() doc at module top.
                if is_trivial_angle(ctx, t1.arg) {
                    continue;
                }

                // Compute the overall scale factor k
                // For sin: k = coeff1 / 3
                // For cos: k = coeff3 / 4
                let scale = match t1.builtin {
                    BuiltinFn::Sin => &t1.coeff / &three,
                    BuiltinFn::Cos => &t3.coeff / &four,
                    _ => continue,
                };

                // Build sin(3θ) or cos(3θ)
                let three_id = ctx.num(3);
                let triple_arg = smart_mul(ctx, three_id, t1.arg);
                let contracted = ctx.call_builtin(t1.builtin, vec![triple_arg]);

                // Apply scale factor
                let one = num_rational::BigRational::from_integer(1.into());
                let neg_one = -&one;
                let scaled = if scale == one {
                    contracted
                } else if scale == neg_one {
                    ctx.add(Expr::Neg(contracted))
                } else {
                    let scale_id = ctx.add(Expr::Number(scale));
                    smart_mul(ctx, scale_id, contracted)
                };

                // If there are exactly 2 signed terms (the matched pair), return directly
                if signed_terms.len() == 2 {
                    let desc = match t1.builtin {
                        BuiltinFn::Sin => "3sin(θ)−4sin³(θ) → sin(3θ)",
                        BuiltinFn::Cos => "4cos³(θ)−3cos(θ) → cos(3θ)",
                        _ => "triple angle contraction",
                    };
                    return Some(Rewrite::new(scaled).desc(desc));
                }

                // N-ary case: reconstruct sum with remaining terms + contracted
                let mut new_terms: Vec<ExprId> = Vec::new();
                for (k, (term, sign)) in signed_terms.iter().enumerate() {
                    if k != t1.index && k != t3.index {
                        if *sign == crate::nary::Sign::Neg {
                            new_terms.push(ctx.add(Expr::Neg(*term)));
                        } else {
                            new_terms.push(*term);
                        }
                    }
                }
                new_terms.push(scaled);

                let mut acc = new_terms[0];
                for &t in new_terms.iter().skip(1) {
                    acc = ctx.add(Expr::Add(acc, t));
                }

                let desc = match t1.builtin {
                    BuiltinFn::Sin => "3sin(θ)−4sin³(θ) → sin(3θ)",
                    BuiltinFn::Cos => "4cos³(θ)−3cos(θ) → cos(3θ)",
                    _ => "triple angle contraction",
                };
                return Some(Rewrite::new(acc).desc(desc));
            }
        }

        None
    }
);
