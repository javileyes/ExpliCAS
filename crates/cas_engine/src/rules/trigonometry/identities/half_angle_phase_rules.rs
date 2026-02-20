//! Half-angle, cotangent half-angle, Weierstrass, and identity zero rules.

use crate::define_rule;
use crate::helpers::{as_add, as_div, as_sub};
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::expr_rewrite::smart_mul;
use cas_math::trig_half_angle_support::{extract_cot_term, extract_tan_half_angle, is_half_angle};
use cas_math::trig_weierstrass_support::{
    match_one_minus_tan_half_squared as match_one_minus_tan_squared,
    match_one_plus_tan_half_squared as match_one_plus_tan_squared,
};
use std::cmp::Ordering;

// ============================================================================
// Cotangent Half-Angle Difference Rule
// ============================================================================
// cot(u/2) - cot(u) = 1/sin(u) = csc(u)
//
// This is a common precalculus identity that avoids term explosion from
// brute-force expansion via cot→cos/sin + double angle formulas.
//
// Pattern matching:
// - cot(u/2) - cot(u) → 1/sin(u)
// - k*cot(u/2) - k*cot(u) → k/sin(u)
// - Works on n-ary sums via flatten_add

// =============================================================================
// WEIERSTRASS HALF-ANGLE TANGENT CONTRACTION RULES
// =============================================================================
// Recognize patterns with t = tan(x/2) and contract to sin(x), cos(x):
// - 2*t / (1 + t²) → sin(x)
// - (1 - t²) / (1 + t²) → cos(x)
// This is the CONTRACTION direction (safe, doesn't worsen expressions)

// Weierstrass Contraction Rule: 2*tan(x/2)/(1+tan²(x/2)) → sin(x)
// and (1-tan²(x/2))/(1+tan²(x/2)) → cos(x)
pub struct WeierstrassContractionRule;

impl crate::rule::Rule for WeierstrassContractionRule {
    fn name(&self) -> &str {
        "Weierstrass Half-Angle Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        // Only match Div nodes
        let (num_id, den_id) = as_div(ctx, expr)?;

        // Pattern 1: 2*tan(x/2) / (1 + tan²(x/2)) → sin(x)
        // Check denominator: 1 + tan²(x/2)
        if let Some((full_angle, tan_half)) = match_one_plus_tan_squared(ctx, den_id) {
            // Check numerator: 2*tan(x/2)
            if let Expr::Mul(l, r) = ctx.get(num_id) {
                let (two_id, tan_id) = if matches!(ctx.get(*l), Expr::Number(n) if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()))
                {
                    (*l, *r)
                } else if matches!(ctx.get(*r), Expr::Number(n) if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()))
                {
                    (*r, *l)
                } else {
                    return self.try_cos_pattern(ctx, num_id, den_id, full_angle, tan_half);
                };
                let _ = two_id;

                // Check if tan_id is tan(x/2) with same argument
                if let Some(tan_arg) = extract_tan_half_angle(ctx, tan_id) {
                    if crate::ordering::compare_expr(ctx, tan_arg, full_angle)
                        == std::cmp::Ordering::Equal
                    {
                        let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![full_angle]);
                        return Some(
                            Rewrite::new(sin_x).desc("2·tan(x/2)/(1 + tan²(x/2)) = sin(x)"),
                        );
                    }
                }
            }

            // Pattern 2: (1 - tan²(x/2)) / (1 + tan²(x/2)) → cos(x)
            return self.try_cos_pattern(ctx, num_id, den_id, full_angle, tan_half);
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

impl WeierstrassContractionRule {
    fn try_cos_pattern(
        &self,
        ctx: &mut cas_ast::Context,
        num_id: ExprId,
        den_id: ExprId,
        _expected_angle: ExprId,
        _expected_tan_half: ExprId,
    ) -> Option<Rewrite> {
        // Pattern 2: (1 - tan²(x/2)) / (1 + tan²(x/2)) → cos(x)
        if let Some((num_angle, _num_tan_half)) = match_one_minus_tan_squared(ctx, num_id) {
            if let Some((den_angle, _den_tan_half)) = match_one_plus_tan_squared(ctx, den_id) {
                // Check angles are the same
                if crate::ordering::compare_expr(ctx, num_angle, den_angle)
                    == std::cmp::Ordering::Equal
                {
                    let cos_x = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![num_angle]);
                    return Some(
                        Rewrite::new(cos_x).desc("(1 - tan²(x/2))/(1 + tan²(x/2)) = cos(x)"),
                    );
                }
            }
        }

        None
    }
}

define_rule!(
    CotHalfAngleDifferenceRule,
    "Cotangent Half-Angle Difference",
    |ctx, expr| {
        // Only match Add or Sub at top level
        // Normalize Sub to Add(a, Neg(b)) conceptually by handling both
        let terms: Vec<ExprId> = if as_add(ctx, expr).is_some() {
            crate::nary::add_leaves(ctx, expr).to_vec()
        } else if let Some((l, r)) = as_sub(ctx, expr) {
            // Treat as [l, -r]
            vec![l, r] // We'll handle the sign in matching
        } else {
            return None;
        };

        if terms.len() < 2 {
            return None;
        }

        // For Sub, we have special handling
        let is_explicit_sub = matches!(ctx.get(expr), Expr::Sub(_, _));

        // Collect cot terms: (index, coeff, arg, is_positive_in_original)
        struct CotTerm {
            index: usize,
            coeff: Option<ExprId>, // None means coefficient is 1
            arg: ExprId,
            is_positive: bool,
        }

        let mut cot_terms = Vec::new();

        if is_explicit_sub {
            // For Sub(a, b): a is positive, b is effectively negative
            if let Some((c1, arg1, _)) = extract_cot_term(ctx, terms[0]) {
                cot_terms.push(CotTerm {
                    index: 0,
                    coeff: c1,
                    arg: arg1,
                    is_positive: true,
                });
            }
            if let Some((c2, arg2, sign2)) = extract_cot_term(ctx, terms[1]) {
                // In Sub(a, b), b appears with flipped sign
                cot_terms.push(CotTerm {
                    index: 1,
                    coeff: c2,
                    arg: arg2,
                    is_positive: !sign2, // Flip because it's subtracted
                });
            }
        } else {
            // For Add chain
            for (i, &term) in terms.iter().enumerate() {
                if let Some((c, arg, is_pos)) = extract_cot_term(ctx, term) {
                    cot_terms.push(CotTerm {
                        index: i,
                        coeff: c,
                        arg,
                        is_positive: is_pos,
                    });
                }
            }
        }

        // Look for pairs: cot(u/2) and cot(u) with opposite signs
        for i in 0..cot_terms.len() {
            for j in 0..cot_terms.len() {
                if i == j {
                    continue;
                }

                let t_half = &cot_terms[i];
                let t_full = &cot_terms[j];

                // Check if t_half.arg is half of t_full.arg
                if let Some(full_angle) = is_half_angle(ctx, t_half.arg) {
                    // Verify full_angle == t_full.arg
                    if crate::ordering::compare_expr(ctx, full_angle, t_full.arg) != Ordering::Equal
                    {
                        continue;
                    }

                    // Check that coefficients match (or both are 1)
                    let coeffs_match = match (&t_half.coeff, &t_full.coeff) {
                        (None, None) => true,
                        (Some(c1), Some(c2)) => {
                            crate::ordering::compare_expr(ctx, *c1, *c2) == Ordering::Equal
                        }
                        _ => false,
                    };

                    if !coeffs_match {
                        continue;
                    }

                    // Check signs: cot(u/2) positive AND cot(u) negative = cot(u/2) - cot(u)
                    // OR cot(u/2) negative AND cot(u) positive = -cot(u/2) + cot(u) = -(cot(u/2) - cot(u))
                    if t_half.is_positive && !t_full.is_positive {
                        // cot(u/2) - cot(u) → 1/sin(u)
                        let one = ctx.num(1);
                        let sin_u = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![t_full.arg]);
                        let result = ctx.add(Expr::Div(one, sin_u));

                        // Apply coefficient if present
                        let final_result = if let Some(c) = t_half.coeff {
                            smart_mul(ctx, c, result)
                        } else {
                            result
                        };

                        // Reconstruct expression without the matched terms
                        if is_explicit_sub && terms.len() == 2 {
                            // Simple case: Sub(cot(u/2), cot(u)) → 1/sin(u)
                            return Some(
                                Rewrite::new(final_result).desc("cot(u/2) - cot(u) = 1/sin(u)"),
                            );
                        }

                        // N-ary case: rebuild sum without matched terms
                        let mut new_terms: Vec<ExprId> = Vec::new();
                        for (k, &term) in terms.iter().enumerate() {
                            if k != t_half.index && k != t_full.index {
                                new_terms.push(term);
                            }
                        }
                        new_terms.push(final_result);

                        let mut new_expr = new_terms[0];
                        for &term in new_terms.iter().skip(1) {
                            new_expr = ctx.add(Expr::Add(new_expr, term));
                        }

                        return Some(Rewrite::new(new_expr).desc("cot(u/2) - cot(u) = 1/sin(u)"));
                    } else if !t_half.is_positive && t_full.is_positive {
                        // -cot(u/2) + cot(u) → -1/sin(u)
                        let one = ctx.num(1);
                        let sin_u = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![t_full.arg]);
                        let result = ctx.add(Expr::Div(one, sin_u));
                        let neg_result = ctx.add(Expr::Neg(result));

                        // Apply coefficient if present
                        let final_result = if let Some(c) = t_half.coeff {
                            smart_mul(ctx, c, neg_result)
                        } else {
                            neg_result
                        };

                        if is_explicit_sub && terms.len() == 2 {
                            return Some(
                                Rewrite::new(final_result).desc("-cot(u/2) + cot(u) = -1/sin(u)"),
                            );
                        }

                        // N-ary case
                        let mut new_terms: Vec<ExprId> = Vec::new();
                        for (k, &term) in terms.iter().enumerate() {
                            if k != t_half.index && k != t_full.index {
                                new_terms.push(term);
                            }
                        }
                        new_terms.push(final_result);

                        let mut new_expr = new_terms[0];
                        for &term in new_terms.iter().skip(1) {
                            new_expr = ctx.add(Expr::Add(new_expr, term));
                        }

                        return Some(Rewrite::new(new_expr).desc("-cot(u/2) + cot(u) = -1/sin(u)"));
                    }
                }
            }
        }

        None
    }
);

// =============================================================================
// TanDifferenceRule: tan(a - b) → (tan(a) - tan(b)) / (1 + tan(a)*tan(b))
// =============================================================================

define_rule!(TanDifferenceRule, "Tangent Difference", |ctx, expr| {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tan)) && args.len() == 1 {
            let arg = args[0];
            // Check if argument is a - b
            if let Expr::Sub(a, b) = ctx.get(arg) {
                let a = *a;
                let b = *b;

                // Build tan(a) - tan(b)
                let tan_a = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![a]);
                let tan_b = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![b]);
                let numerator = ctx.add(Expr::Sub(tan_a, tan_b));

                // Build 1 + tan(a)*tan(b)
                let tan_a2 = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![a]);
                let tan_b2 = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![b]);
                let product = ctx.add(Expr::Mul(tan_a2, tan_b2));
                let one = ctx.num(1);
                let denominator = ctx.add(Expr::Add(one, product));

                let result = ctx.add(Expr::Div(numerator, denominator));
                return Some(
                    Rewrite::new(result).desc("tan(a-b) = (tan(a)-tan(b))/(1+tan(a)·tan(b))"),
                );
            }
        }
    }
    None
});

// =============================================================================
// HyperbolicTanhPythRule: 1 - tanh(x)² → 1/cosh(x)² (sech²)
// =============================================================================
// Canonical direction: contract to reciprocal form.

define_rule!(
    HyperbolicTanhPythRule,
    "Hyperbolic Tanh Pythagorean",
    |ctx, expr| {
        // Flatten the additive chain
        let terms = crate::nary::add_leaves(ctx, expr);

        if terms.len() < 2 {
            return None;
        }

        // Look for pattern: 1 + (-tanh²(x)) i.e. 1 - tanh²(x)
        let mut one_idx: Option<usize> = None;
        let mut tanh2_idx: Option<usize> = None;
        let mut tanh_arg: Option<ExprId> = None;
        let mut is_negative_tanh2 = false;

        for (i, &term) in terms.iter().enumerate() {
            // Check for literal 1
            if let Expr::Number(n) = ctx.get(term) {
                if *n == num_rational::BigRational::from_integer(1.into()) {
                    one_idx = Some(i);
                    continue;
                }
            }
            // Check for -tanh²(x)
            if let Expr::Neg(inner) = ctx.get(term) {
                if let Expr::Pow(base, exp) = ctx.get(*inner) {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        if *n == num_rational::BigRational::from_integer(2.into()) {
                            if let Expr::Function(fn_id, args) = ctx.get(*base) {
                                if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tanh))
                                    && args.len() == 1
                                {
                                    tanh2_idx = Some(i);
                                    tanh_arg = Some(args[0]);
                                    is_negative_tanh2 = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // If we found 1 and -tanh²(x), replace with 1/cosh²(x)
        if let (Some(one_i), Some(tanh_i), Some(arg)) = (one_idx, tanh2_idx, tanh_arg) {
            if is_negative_tanh2 {
                let cosh_func = ctx.call_builtin(cas_ast::BuiltinFn::Cosh, vec![arg]);
                let two = ctx.num(2);
                let cosh_squared = ctx.add(Expr::Pow(cosh_func, two));
                let one = ctx.num(1);
                let sech_squared = ctx.add(Expr::Div(one, cosh_squared));

                // Build new expression
                let mut new_terms: Vec<ExprId> = Vec::new();
                for (j, &t) in terms.iter().enumerate() {
                    if j != one_i && j != tanh_i {
                        new_terms.push(t);
                    }
                }
                new_terms.push(sech_squared);

                let result = if new_terms.len() == 1 {
                    new_terms[0]
                } else {
                    let mut acc = new_terms[0];
                    for &t in new_terms.iter().skip(1) {
                        acc = ctx.add(Expr::Add(acc, t));
                    }
                    acc
                };

                return Some(Rewrite::new(result).desc("1 - tanh²(x) = 1/cosh²(x)"));
            }
        }

        None
    }
);

// =============================================================================
// HyperbolicHalfAngleSquaresRule: cosh(x/2)² → (cosh(x)+1)/2, sinh(x/2)² → (cosh(x)-1)/2
