//! Identity zero rules: pattern-driven cancellation for trig identities.
//!
//! Extracted from `half_angle_phase_rules.rs` to reduce file size.
//! Contains:
//! - WeierstrassSinIdentityZeroRule: sin(x) - 2*tan(x/2)/(1+tan²(x/2)) → 0
//! - WeierstrassCosIdentityZeroRule: cos(x) - (1-tan²(x/2))/(1+tan²(x/2)) → 0
//! - Sin4xIdentityZeroRule: sin(4t) - 4*sin(t)*cos(t)*(cos²(t)-sin²(t)) → 0
//! - TanDifferenceIdentityZeroRule: tan(a-b) - (tan(a)-tan(b))/(1+tan(a)*tan(b)) → 0

use crate::helpers::{as_add, as_neg, as_sub};
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::trig_identity_zero_support::{
    is_cos_squared_t, is_sin_squared_t, match_one_plus_tan_product,
};
use cas_math::trig_weierstrass_support::{
    match_one_minus_tan_half_squared, match_one_plus_tan_half_squared, match_two_tan_half,
};

// =============================================================================
// WEIERSTRASS IDENTITY ZERO RULES (Pattern-Driven Cancellation)
// =============================================================================
// These rules detect the complete Weierstrass identity patterns and cancel to 0
// directly, avoiding explosive expansion through tan→sin/cos conversion.
//
// sin(x) - 2*tan(x/2)/(1 + tan(x/2)²) → 0
// cos(x) - (1 - tan(x/2)²)/(1 + tan(x/2)²) → 0

// WeierstrassSinIdentityZeroRule: sin(x) - 2*tan(x/2)/(1+tan²(x/2)) → 0
// Pattern-driven cancellation, no expansion.
pub struct WeierstrassSinIdentityZeroRule;

impl crate::rule::Rule for WeierstrassSinIdentityZeroRule {
    fn name(&self) -> &str {
        "Weierstrass Sin Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only match Sub nodes: sin(x) - RHS or RHS - sin(x)
        let (left, right, negated) = if let Some((l, r)) = as_sub(ctx, expr) {
            (l, r, false)
        } else if let Some((l, r)) = as_add(ctx, expr) {
            // Check if one side is negated
            if let Some(inner) = as_neg(ctx, r) {
                (l, inner, false)
            } else if let Some(inner) = as_neg(ctx, l) {
                (r, inner, false)
            } else {
                return None;
            }
        } else {
            return None;
        };
        let _ = negated;

        // Try both orderings: sin(x) - RHS and RHS - sin(x)
        if let Some(result) = self.try_match(ctx, left, right) {
            return Some(result);
        }
        if let Some(result) = self.try_match(ctx, right, left) {
            return Some(result);
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200 // Very high - must run BEFORE Pythagorean 1+tan²→sec²
    }
}

impl WeierstrassSinIdentityZeroRule {
    /// Try to match sin(x) = left, RHS = right
    fn try_match(
        &self,
        ctx: &mut cas_ast::Context,
        sin_side: ExprId,
        rhs: ExprId,
    ) -> Option<Rewrite> {
        // Check if sin_side is sin(x)
        if let Expr::Function(fn_id, args) = ctx.get(sin_side) {
            if !matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sin)) || args.len() != 1 {
                return None;
            }
            let full_angle = args[0];

            // Check if rhs is 2*tan(x/2) / (1 + tan²(x/2))
            if let Expr::Div(num, den) = ctx.get(rhs) {
                // Numerator: 2*tan(x/2)
                if let Some(num_angle) = match_two_tan_half(ctx, *num) {
                    // Denominator: 1 + tan²(x/2)
                    if let Some((den_angle, _)) = match_one_plus_tan_half_squared(ctx, *den) {
                        // Check all angles match
                        if crate::ordering::compare_expr(ctx, full_angle, num_angle)
                            == std::cmp::Ordering::Equal
                            && crate::ordering::compare_expr(ctx, full_angle, den_angle)
                                == std::cmp::Ordering::Equal
                        {
                            let zero = ctx.num(0);
                            return Some(
                                Rewrite::new(zero)
                                    .desc("sin(x) = 2·tan(x/2)/(1 + tan²(x/2)) [Weierstrass]"),
                            );
                        }
                    }
                }
            }
        }
        None
    }
}

// WeierstrassCosIdentityZeroRule: cos(x) - (1-tan²(x/2))/(1+tan²(x/2)) → 0
pub struct WeierstrassCosIdentityZeroRule;

impl crate::rule::Rule for WeierstrassCosIdentityZeroRule {
    fn name(&self) -> &str {
        "Weierstrass Cos Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only match Sub nodes: cos(x) - RHS or RHS - cos(x)
        let (left, right) = if let Some((l, r)) = as_sub(ctx, expr) {
            (l, r)
        } else if let Some((l, r)) = as_add(ctx, expr) {
            // Check if one side is negated
            if let Some(inner) = as_neg(ctx, r) {
                (l, inner)
            } else if let Some(inner) = as_neg(ctx, l) {
                (r, inner)
            } else {
                return None;
            }
        } else {
            return None;
        };

        // Try both orderings
        if let Some(result) = self.try_match(ctx, left, right) {
            return Some(result);
        }
        if let Some(result) = self.try_match(ctx, right, left) {
            return Some(result);
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200 // Very high - must run BEFORE Pythagorean 1+tan²→sec²
    }
}

impl WeierstrassCosIdentityZeroRule {
    /// Try to match cos(x) = cos_side, RHS = rhs
    fn try_match(
        &self,
        ctx: &mut cas_ast::Context,
        cos_side: ExprId,
        rhs: ExprId,
    ) -> Option<Rewrite> {
        // Check if cos_side is cos(x)
        if let Expr::Function(fn_id, args) = ctx.get(cos_side) {
            if !matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) || args.len() != 1 {
                return None;
            }
            let full_angle = args[0];

            // Check if rhs is (1 - tan²(x/2)) / (1 + tan²(x/2))
            if let Expr::Div(num, den) = ctx.get(rhs) {
                // Numerator: 1 - tan²(x/2)
                if let Some((num_angle, _)) = match_one_minus_tan_half_squared(ctx, *num) {
                    // Denominator: 1 + tan²(x/2)
                    if let Some((den_angle, _)) = match_one_plus_tan_half_squared(ctx, *den) {
                        // Check all angles match
                        if crate::ordering::compare_expr(ctx, full_angle, num_angle)
                            == std::cmp::Ordering::Equal
                            && crate::ordering::compare_expr(ctx, full_angle, den_angle)
                                == std::cmp::Ordering::Equal
                        {
                            let zero = ctx.num(0);
                            return Some(
                                Rewrite::new(zero)
                                    .desc("cos(x) = (1 - tan²(x/2))/(1 + tan²(x/2)) [Weierstrass]"),
                            );
                        }
                    }
                }
            }
        }
        None
    }
}

// =============================================================================
// Sin4xIdentityZeroRule: sin(4t) - 4*sin(t)*cos(t)*(cos²(t)-sin²(t)) → 0
// =============================================================================
// Recognizes the sin(4x) expansion identity directly in cancellation context.

pub struct Sin4xIdentityZeroRule;

impl crate::rule::Rule for Sin4xIdentityZeroRule {
    fn name(&self) -> &str {
        "Sin 4x Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only match Sub nodes or Add with negated term
        let (left, right) = if let Some((l, r)) = as_sub(ctx, expr) {
            (l, r)
        } else if let Some((l, r)) = as_add(ctx, expr) {
            if let Some(inner) = as_neg(ctx, r) {
                (l, inner)
            } else if let Some(inner) = as_neg(ctx, l) {
                (r, inner)
            } else {
                return None;
            }
        } else {
            return None;
        };

        // Try both orderings
        if let Some(result) = self.try_match(ctx, left, right) {
            return Some(result);
        }
        if let Some(result) = self.try_match(ctx, right, left) {
            return Some(result);
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }

    fn priority(&self) -> i32 {
        200 // Run before expansion
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

impl Sin4xIdentityZeroRule {
    fn try_match(&self, ctx: &mut cas_ast::Context, lhs: ExprId, rhs: ExprId) -> Option<Rewrite> {
        // LHS should be sin(4*t)
        if let Expr::Function(fn_id, args) = ctx.get(lhs) {
            if !matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sin)) || args.len() != 1 {
                return None;
            }
            let sin_arg = args[0];

            // Check if arg is 4*t
            let t = match ctx.get(sin_arg) {
                Expr::Mul(l, r) => {
                    // Check for Mul(4, t) or Mul(t, 4)
                    let l = *l;
                    let r = *r;
                    if let Expr::Number(n) = ctx.get(l) {
                        if *n == num_rational::BigRational::from_integer(4.into()) {
                            r
                        } else {
                            return None;
                        }
                    } else if let Expr::Number(n) = ctx.get(r) {
                        if *n == num_rational::BigRational::from_integer(4.into()) {
                            l
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                _ => return None,
            };

            // RHS should be 4*sin(t)*cos(t)*(cos(t)^2 - sin(t)^2)
            // Flatten multiplication to get factors
            let factors = crate::nary::mul_leaves(ctx, rhs);

            if factors.len() < 4 {
                return None;
            }

            // Look for: 4, sin(t), cos(t), (cos(t)^2 - sin(t)^2) or cos(2t)
            let mut has_four = false;
            let mut has_sin_t = false;
            let mut has_cos_t = false;
            let mut has_diff_squares = false;

            for &factor in &factors {
                // Check for 4
                if let Expr::Number(n) = ctx.get(factor) {
                    if *n == num_rational::BigRational::from_integer(4.into()) {
                        has_four = true;
                        continue;
                    }
                }
                // Check for sin(t)
                if let Expr::Function(fn_name, fn_args) = ctx.get(factor) {
                    if ctx.is_builtin(*fn_name, BuiltinFn::Sin)
                        && fn_args.len() == 1
                        && crate::ordering::compare_expr(ctx, fn_args[0], t)
                            == std::cmp::Ordering::Equal
                    {
                        has_sin_t = true;
                        continue;
                    }
                    if ctx.is_builtin(*fn_name, BuiltinFn::Cos) && fn_args.len() == 1 {
                        let arg = fn_args[0];
                        if crate::ordering::compare_expr(ctx, arg, t) == std::cmp::Ordering::Equal {
                            has_cos_t = true;
                            continue;
                        }
                        // Also check for cos(2t) which equals cos²-sin²
                        if let Expr::Mul(cl, cr) = ctx.get(arg) {
                            let cl = *cl;
                            let cr = *cr;
                            let is_2t = if let Expr::Number(n) = ctx.get(cl) {
                                *n == num_rational::BigRational::from_integer(2.into())
                                    && crate::ordering::compare_expr(ctx, cr, t)
                                        == std::cmp::Ordering::Equal
                            } else if let Expr::Number(n) = ctx.get(cr) {
                                *n == num_rational::BigRational::from_integer(2.into())
                                    && crate::ordering::compare_expr(ctx, cl, t)
                                        == std::cmp::Ordering::Equal
                            } else {
                                false
                            };
                            if is_2t {
                                has_diff_squares = true;
                                continue;
                            }
                        }
                    }
                }
                // Check for cos²(t) - sin²(t)
                if let Expr::Sub(sl, sr) = ctx.get(factor) {
                    let sl = *sl;
                    let sr = *sr;
                    if is_cos_squared_t(ctx, sl, t) && is_sin_squared_t(ctx, sr, t) {
                        has_diff_squares = true;
                        continue;
                    }
                }
            }

            if has_four && has_sin_t && has_cos_t && has_diff_squares {
                let zero = ctx.num(0);
                return Some(
                    Rewrite::new(zero).desc("sin(4t) = 4·sin(t)·cos(t)·(cos²(t)-sin²(t))"),
                );
            }
        }
        None
    }
}

// =============================================================================
// TanDifferenceIdentityZeroRule: tan(a-b) - (tan(a)-tan(b))/(1+tan(a)*tan(b)) → 0
// =============================================================================
// Recognizes the tangent difference identity directly in cancellation context.

pub struct TanDifferenceIdentityZeroRule;

impl crate::rule::Rule for TanDifferenceIdentityZeroRule {
    fn name(&self) -> &str {
        "Tangent Difference Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only match Sub nodes or Add with negated term
        let (left, right, _negated) = if let Some((l, r)) = as_sub(ctx, expr) {
            (l, r, false)
        } else if let Some((l, r)) = as_add(ctx, expr) {
            if let Some(inner) = as_neg(ctx, r) {
                (l, inner, true)
            } else if let Some(inner) = as_neg(ctx, l) {
                (r, inner, true)
            } else {
                return None;
            }
        } else {
            return None;
        };

        // Try both orderings: tan(a-b) - RHS or RHS - tan(a-b)
        if let Some(result) = self.try_match(ctx, left, right) {
            return Some(result);
        }
        if let Some(result) = self.try_match(ctx, right, left) {
            return Some(result);
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }

    fn priority(&self) -> i32 {
        200 // Run before tan→sin/cos expansion
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

impl TanDifferenceIdentityZeroRule {
    fn try_match(&self, ctx: &mut cas_ast::Context, lhs: ExprId, rhs: ExprId) -> Option<Rewrite> {
        // LHS should be tan(a - b)
        if let Expr::Function(fn_id, args) = ctx.get(lhs) {
            if !matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tan)) || args.len() != 1 {
                return None;
            }
            let tan_arg = args[0];

            // Extract a and b from (a - b) or (a + (-b))
            let (a, b) = match ctx.get(tan_arg) {
                Expr::Sub(l, r) => (*l, *r),
                Expr::Add(l, r) => {
                    if let Expr::Neg(inner) = ctx.get(*r) {
                        (*l, *inner)
                    } else if let Expr::Neg(inner) = ctx.get(*l) {
                        (*r, *inner)
                    } else {
                        return None;
                    }
                }
                _ => return None,
            };

            // RHS should be (tan(a) - tan(b)) / (1 + tan(a)*tan(b))
            if let Expr::Div(num, den) = ctx.get(rhs) {
                let num = *num;
                let den = *den;

                // Check numerator: tan(a) - tan(b)
                let (tan_a_num, tan_b_num) = match ctx.get(num) {
                    Expr::Sub(l, r) => (*l, *r),
                    Expr::Add(l, r) => {
                        if let Expr::Neg(inner) = ctx.get(*r) {
                            (*l, *inner)
                        } else {
                            return None;
                        }
                    }
                    _ => return None,
                };

                // Verify tan_a_num is tan(a)
                if let Expr::Function(name_a, args_a) = ctx.get(tan_a_num) {
                    if !ctx.is_builtin(*name_a, BuiltinFn::Tan) || args_a.len() != 1 {
                        return None;
                    }
                    if crate::ordering::compare_expr(ctx, args_a[0], a) != std::cmp::Ordering::Equal
                    {
                        return None;
                    }
                } else {
                    return None;
                }

                // Verify tan_b_num is tan(b)
                if let Expr::Function(name_b, args_b) = ctx.get(tan_b_num) {
                    if !ctx.is_builtin(*name_b, BuiltinFn::Tan) || args_b.len() != 1 {
                        return None;
                    }
                    if crate::ordering::compare_expr(ctx, args_b[0], b) != std::cmp::Ordering::Equal
                    {
                        return None;
                    }
                } else {
                    return None;
                }

                // Check denominator: 1 + tan(a)*tan(b)
                if !match_one_plus_tan_product(ctx, den, a, b) {
                    return None;
                }

                // All matched! Return 0
                let zero = ctx.num(0);
                return Some(
                    Rewrite::new(zero).desc("tan(a-b) = (tan(a)-tan(b))/(1+tan(a)·tan(b))"),
                );
            }
        }
        None
    }
}
