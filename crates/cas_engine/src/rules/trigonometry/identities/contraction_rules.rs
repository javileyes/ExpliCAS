//! Contraction rules for trigonometric expressions.
//!
//! These are the INVERSE of expansion rules — they contract expanded forms back
//! to compact representations (half-angle tangent, double angle contraction).

use crate::helpers::{as_div, as_mul, as_neg, as_sub};
use crate::nary::Sign;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::trig_sum_product_support::extract_trig_arg as extract_trig_arg_named;

// =============================================================================
// HALF-ANGLE TANGENT RULE
// (1 - cos(2x)) / sin(2x) → tan(x)
// sin(2x) / (1 + cos(2x)) → tan(x)
// =============================================================================
// These are half-angle tangent identities derived from:
//   1 - cos(2x) = 2·sin²(x)
//   1 + cos(2x) = 2·cos²(x)
//   sin(2x) = 2·sin(x)·cos(x)
//
// DOMAIN WARNING: This transformation can extend the domain:
// - Pattern 1: Original requires sin(2x) ≠ 0, but tan(x) only requires cos(x) ≠ 0
// - Pattern 2: Original requires 1+cos(2x) ≠ 0, but tan(x) only requires cos(x) ≠ 0
//
// To preserve soundness, we introduce requires for cos(x) ≠ 0 (for tan(x) to be defined)
// and inherit the original denominator ≠ 0 condition.
//
// Uses SoundnessLabel::EquivalenceUnderIntroducedRequires
pub struct HalfAngleTangentRule;

impl crate::rule::Rule for HalfAngleTangentRule {
    fn name(&self) -> &str {
        "Half-Angle Tangent Identity"
    }

    fn priority(&self) -> i32 {
        50 // Normal priority
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::extract_double_angle_arg;
        use crate::implicit_domain::ImplicitCondition;

        // Only match Div nodes
        let (num_id, den_id) = as_div(ctx, expr)?;

        // Pattern 1: (1 - cos(2x)) / sin(2x) → tan(x)
        // Pattern 2: sin(2x) / (1 + cos(2x)) → tan(x)

        enum Pattern {
            OneMinusCosOverSin { x: ExprId, sin_2x: ExprId },
            SinOverOnePlusCos { x: ExprId, one_plus_cos_2x: ExprId },
        }

        let pattern = 'pattern: {
            // Try Pattern 1: (1 - cos(2x)) / sin(2x)
            // Numerator can be: Sub(1, cos(2x)) OR Add(1, Neg(cos(2x))) (canonicalized)

            // Helper to extract cos(2x) from either cos(2x) or Neg(cos(2x))
            let try_extract_cos_2x = |ctx: &cas_ast::Context,
                                      id: ExprId|
             -> Option<(ExprId, bool)> {
                if let Expr::Function(fn_id, args) = ctx.get(id) {
                    if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) && args.len() == 1 {
                        return extract_double_angle_arg(ctx, args[0]).map(|x| (x, false));
                    }
                }
                // Check for Neg(cos(2x))
                if let Expr::Neg(inner) = ctx.get(id) {
                    if let Expr::Function(fn_id, args) = ctx.get(*inner) {
                        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) && args.len() == 1
                        {
                            return extract_double_angle_arg(ctx, args[0]).map(|x| (x, true));
                            // negated
                        }
                    }
                }
                None
            };

            // Check Sub(1, cos(2x))
            if let Expr::Sub(one_id, cos_id) = ctx.get(num_id) {
                if let Expr::Number(n) = ctx.get(*one_id) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                        if let Some((x, false)) = try_extract_cos_2x(ctx, *cos_id) {
                            // Check if denominator is sin(2x) with same argument
                            if let Expr::Function(den_fn_id, den_args) = ctx.get(den_id) {
                                if matches!(ctx.builtin_of(*den_fn_id), Some(BuiltinFn::Sin))
                                    && den_args.len() == 1
                                {
                                    if let Some(x2) = extract_double_angle_arg(ctx, den_args[0]) {
                                        if crate::ordering::compare_expr(ctx, x, x2)
                                            == std::cmp::Ordering::Equal
                                        {
                                            break 'pattern Some(Pattern::OneMinusCosOverSin {
                                                x,
                                                sin_2x: den_id,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Check Add(1, Neg(cos(2x))) or Add(Neg(cos(2x)), 1) - canonicalized form
            if let Expr::Add(left, right) = ctx.get(num_id) {
                // Try left=1, right=Neg(cos)
                let try_order = |one: ExprId, neg_cos: ExprId| -> Option<ExprId> {
                    if let Expr::Number(n) = ctx.get(one) {
                        if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into())
                        {
                            if let Some((x, true)) = try_extract_cos_2x(ctx, neg_cos) {
                                return Some(x);
                            }
                        }
                    }
                    None
                };

                // Try both orders
                let x_opt = try_order(*left, *right).or_else(|| try_order(*right, *left));

                if let Some(x) = x_opt {
                    // Check if denominator is sin(2x) with same argument
                    if let Expr::Function(den_fn_id, den_args) = ctx.get(den_id) {
                        if matches!(ctx.builtin_of(*den_fn_id), Some(BuiltinFn::Sin))
                            && den_args.len() == 1
                        {
                            if let Some(x2) = extract_double_angle_arg(ctx, den_args[0]) {
                                if crate::ordering::compare_expr(ctx, x, x2)
                                    == std::cmp::Ordering::Equal
                                {
                                    break 'pattern Some(Pattern::OneMinusCosOverSin {
                                        x,
                                        sin_2x: den_id,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // Try Pattern 2: sin(2x) / (1 + cos(2x))
            // Numerator: sin(2x)
            if let Expr::Function(fn_id, args) = ctx.get(num_id) {
                if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sin)) && args.len() == 1 {
                    if let Some(x) = extract_double_angle_arg(ctx, args[0]) {
                        // Denominator: 1 + cos(2x) or Add(1, cos(2x))
                        if let Expr::Add(left, right) = ctx.get(den_id) {
                            // Check both orders: 1 + cos(2x) or cos(2x) + 1
                            let (one_id, cos_id) = if matches!(ctx.get(*left), Expr::Number(n) if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()))
                            {
                                (*left, *right)
                            } else if matches!(ctx.get(*right), Expr::Number(n) if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()))
                            {
                                (*right, *left)
                            } else {
                                break 'pattern None;
                            };

                            // Verify one_id is 1
                            if let Expr::Number(n) = ctx.get(one_id) {
                                if n.is_integer()
                                    && *n == num_rational::BigRational::from_integer(1.into())
                                {
                                    // Check if cos_id is cos(2x) with same x
                                    if let Expr::Function(cos_fn_id, cos_args) = ctx.get(cos_id) {
                                        if matches!(
                                            ctx.builtin_of(*cos_fn_id),
                                            Some(BuiltinFn::Cos)
                                        ) && cos_args.len() == 1
                                        {
                                            if let Some(x2) =
                                                extract_double_angle_arg(ctx, cos_args[0])
                                            {
                                                if crate::ordering::compare_expr(ctx, x, x2)
                                                    == std::cmp::Ordering::Equal
                                                {
                                                    break 'pattern Some(
                                                        Pattern::SinOverOnePlusCos {
                                                            x,
                                                            one_plus_cos_2x: den_id,
                                                        },
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            None
        }?;

        // Build tan(x)
        let (x, denom_expr, desc) = match pattern {
            Pattern::OneMinusCosOverSin { x, sin_2x } => {
                (x, sin_2x, "(1 - cos(2x))/sin(2x) = tan(x)")
            }
            Pattern::SinOverOnePlusCos { x, one_plus_cos_2x } => {
                (x, one_plus_cos_2x, "sin(2x)/(1 + cos(2x)) = tan(x)")
            }
        };

        let tan_x = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![x]);

        // Build cos(x) for the NonZero require
        let cos_x = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![x]);

        // Create rewrite with requires:
        // 1. Original denominator ≠ 0 (inherited from the division)
        // 2. cos(x) ≠ 0 (for tan(x) to be defined)
        let rewrite = Rewrite::new(tan_x)
            .desc(desc)
            .requires(ImplicitCondition::NonZero(denom_expr))
            .requires(ImplicitCondition::NonZero(cos_x));

        Some(rewrite)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn soundness(&self) -> crate::rule::SoundnessLabel {
        crate::rule::SoundnessLabel::EquivalenceUnderIntroducedRequires
    }
}

// =============================================================================
// DOUBLE ANGLE CONTRACTION RULE
// 2·sin(t)·cos(t) → sin(2t), cos²(t) - sin²(t) → cos(2t)
// =============================================================================
// This is the INVERSE of DoubleAngleRule - contracts expanded forms back to double angle.
// Essential for recognizing Weierstrass substitution identities.
pub struct DoubleAngleContractionRule;

impl crate::rule::Rule for DoubleAngleContractionRule {
    fn name(&self) -> &str {
        "Double Angle Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // GUARD: Don't contract when sin(4x) identity pattern is detected
        // This preserves 4*sin*cos*(cos²-sin²) for Sin4xIdentityZeroRule
        if let Some(marks) = _parent_ctx.pattern_marks() {
            if marks.has_sin4x_identity_pattern {
                return None;
            }
        }

        // Pattern 1: 2·sin(t)·cos(t) → sin(2t)
        // Matches Mul(2, Mul(sin(t), cos(t))) or Mul(Mul(2, sin(t)), cos(t)) etc.
        if let Some((l, r)) = as_mul(ctx, expr) {
            if let Some((sin_arg, cos_arg)) = self.extract_two_sin_cos(ctx, l, r) {
                // Check if sin and cos have the same argument
                if crate::ordering::compare_expr(ctx, sin_arg, cos_arg) == std::cmp::Ordering::Equal
                {
                    // Build sin(2*t)
                    let two = ctx.num(2);
                    let double_arg = ctx.add(Expr::Mul(two, sin_arg));
                    let sin_2t = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![double_arg]);
                    return Some(Rewrite::new(sin_2t).desc("2·sin(t)·cos(t) = sin(2t)"));
                }
            }
        }

        // Pattern 2: cos²(t) - sin²(t) → cos(2t)
        if let Some((l, r)) = as_sub(ctx, expr) {
            if let Some((cos_arg, sin_arg)) = self.extract_cos2_minus_sin2(ctx, l, r) {
                // Check if cos² and sin² have the same argument
                if crate::ordering::compare_expr(ctx, cos_arg, sin_arg) == std::cmp::Ordering::Equal
                {
                    // Build cos(2*t)
                    let two = ctx.num(2);
                    let double_arg = ctx.add(Expr::Mul(two, cos_arg));
                    let cos_2t = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![double_arg]);
                    return Some(Rewrite::new(cos_2t).desc("cos²(t) - sin²(t) = cos(2t)"));
                }
            }
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL.union(crate::target_kind::TargetKindSet::SUB))
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200 // Run before expansion rules to prevent ping-pong
    }
}

impl DoubleAngleContractionRule {
    /// Extract (sin_arg, cos_arg) from 2·sin(t)·cos(t) pattern
    fn extract_two_sin_cos(
        &self,
        ctx: &cas_ast::Context,
        l: ExprId,
        r: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // Check all possible arrangements of 2, sin(t), cos(t)
        let two_rat = num_rational::BigRational::from_integer(2.into());

        // Case: Mul(2, Mul(sin, cos))
        if let Expr::Number(n) = ctx.get(l) {
            if *n == two_rat {
                if let Expr::Mul(a, b) = ctx.get(r) {
                    return self.extract_sin_cos_pair(ctx, *a, *b);
                }
            }
        }

        // Case: Mul(Mul(...), 2)
        if let Expr::Number(n) = ctx.get(r) {
            if *n == two_rat {
                if let Expr::Mul(a, b) = ctx.get(l) {
                    return self.extract_sin_cos_pair(ctx, *a, *b);
                }
            }
        }

        // Case: Mul(Mul(2, sin), cos) or Mul(Mul(2, cos), sin)
        if let Expr::Mul(inner_l, inner_r) = ctx.get(l) {
            if let Expr::Number(n) = ctx.get(*inner_l) {
                if *n == two_rat {
                    // inner_r is either sin or cos
                    return self.extract_trig_and_match(ctx, *inner_r, r);
                }
            }
            if let Expr::Number(n) = ctx.get(*inner_r) {
                if *n == two_rat {
                    return self.extract_trig_and_match(ctx, *inner_l, r);
                }
            }
        }

        // Case: Mul(sin, Mul(2, cos)) or similar
        if let Expr::Mul(inner_l, inner_r) = ctx.get(r) {
            if let Expr::Number(n) = ctx.get(*inner_l) {
                if *n == two_rat {
                    return self.extract_trig_and_match(ctx, *inner_r, l);
                }
            }
            if let Expr::Number(n) = ctx.get(*inner_r) {
                if *n == two_rat {
                    return self.extract_trig_and_match(ctx, *inner_l, l);
                }
            }
        }

        None
    }

    fn extract_sin_cos_pair(
        &self,
        ctx: &cas_ast::Context,
        a: ExprId,
        b: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // Check if a is sin and b is cos, or vice versa
        if let Expr::Function(fn_id_a, args_a) = ctx.get(a) {
            if let Expr::Function(fn_id_b, args_b) = ctx.get(b) {
                if args_a.len() == 1 && args_b.len() == 1 {
                    let builtin_a = ctx.builtin_of(*fn_id_a);
                    let builtin_b = ctx.builtin_of(*fn_id_b);
                    if matches!(builtin_a, Some(BuiltinFn::Sin))
                        && matches!(builtin_b, Some(BuiltinFn::Cos))
                    {
                        return Some((args_a[0], args_b[0]));
                    }
                    if matches!(builtin_a, Some(BuiltinFn::Cos))
                        && matches!(builtin_b, Some(BuiltinFn::Sin))
                    {
                        return Some((args_b[0], args_a[0]));
                    }
                }
            }
        }
        None
    }

    fn extract_trig_and_match(
        &self,
        ctx: &cas_ast::Context,
        trig1: ExprId,
        trig2: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        if let Expr::Function(fn_id1, args1) = ctx.get(trig1) {
            if let Expr::Function(fn_id2, args2) = ctx.get(trig2) {
                if args1.len() == 1 && args2.len() == 1 {
                    let builtin1 = ctx.builtin_of(*fn_id1);
                    let builtin2 = ctx.builtin_of(*fn_id2);
                    if matches!(builtin1, Some(BuiltinFn::Sin))
                        && matches!(builtin2, Some(BuiltinFn::Cos))
                    {
                        return Some((args1[0], args2[0]));
                    }
                    if matches!(builtin1, Some(BuiltinFn::Cos))
                        && matches!(builtin2, Some(BuiltinFn::Sin))
                    {
                        return Some((args2[0], args1[0]));
                    }
                }
            }
        }
        None
    }

    /// Extract (cos_arg, sin_arg) from cos²(t) - sin²(t) pattern
    fn extract_cos2_minus_sin2(
        &self,
        ctx: &cas_ast::Context,
        l: ExprId,
        r: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // l should be cos²(t), r should be sin²(t)
        let two_rat = num_rational::BigRational::from_integer(2.into());

        if let Expr::Pow(base_l, exp_l) = ctx.get(l) {
            if let Expr::Number(n) = ctx.get(*exp_l) {
                if *n == two_rat {
                    if let Expr::Function(fn_id_l, args_l) = ctx.get(*base_l) {
                        if matches!(ctx.builtin_of(*fn_id_l), Some(BuiltinFn::Cos))
                            && args_l.len() == 1
                        {
                            // Check r is sin²
                            if let Expr::Pow(base_r, exp_r) = ctx.get(r) {
                                if let Expr::Number(m) = ctx.get(*exp_r) {
                                    if *m == two_rat {
                                        if let Expr::Function(fn_id_r, args_r) = ctx.get(*base_r) {
                                            if matches!(
                                                ctx.builtin_of(*fn_id_r),
                                                Some(BuiltinFn::Sin)
                                            ) && args_r.len() == 1
                                            {
                                                return Some((args_l[0], args_r[0]));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

// =============================================================================
// Cos2xAdditiveContractionRule: 1 - 2·sin²(t) → cos(2t), 2·cos²(t) - 1 → cos(2t)
// =============================================================================
// These are alternate forms of the double-angle cosine identity that the
// existing DoubleAngleContractionRule does not handle (it only handles
// cos²(t) - sin²(t) → cos(2t)).
//
// Mathematical identities:
//   cos(2t) = 1 - 2·sin²(t)
//   cos(2t) = 2·cos²(t) - 1
//
// We scan additive leaves for a pair: constant ±1 and ∓2·trig²(t).

pub struct Cos2xAdditiveContractionRule;

impl crate::rule::Rule for Cos2xAdditiveContractionRule {
    fn name(&self) -> &str {
        "Cos 2x Additive Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only apply to Add/Sub expressions
        if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
            return None;
        }

        // Use add_terms_signed to properly decompose both Add and Sub nodes.
        // add_leaves only flattens Add, so Sub(2cos²t, 1) would be a single
        // term and never match. add_terms_signed decomposes it into
        // [(2cos²t, +), (1, -)].
        let signed_terms = crate::nary::add_terms_signed(ctx, expr);

        // Only apply to exactly 2-term additive expressions to avoid
        // disrupting larger Pythagorean chains where the contraction would
        // prevent downstream symbolic cancellation.
        if signed_terms.len() != 2 {
            return None;
        }

        let one_rat = num_rational::BigRational::from_integer(1.into());
        let two_rat = num_rational::BigRational::from_integer(2.into());
        let neg_two_rat = num_rational::BigRational::from_integer((-2).into());

        // Find a constant ±1 term (sign is tracked by the signed_terms flag)
        for (i, &(term_i, sign_i)) in signed_terms.iter().enumerate() {
            let term_val = match ctx.get(term_i) {
                Expr::Number(n) => {
                    if sign_i == Sign::Pos {
                        n.clone()
                    } else {
                        -n.clone()
                    }
                }
                _ => continue,
            };

            // Check if this is +1 or -1 (after accounting for sign)
            let is_pos_one = term_val == one_rat;
            let is_neg_one = term_val == -one_rat.clone();
            if !is_pos_one && !is_neg_one {
                continue;
            }

            // Look for a matching ±2·trig²(t) in the remaining terms
            for (j, &(term_j, sign_j)) in signed_terms.iter().enumerate() {
                if j == i {
                    continue;
                }

                if let Some((trig_arg, trig_is_sin, mut coeff)) =
                    Self::extract_coeff_trig_squared(ctx, term_j)
                {
                    // Account for the additive sign from the signed decomposition
                    if sign_j == Sign::Neg {
                        coeff = -coeff;
                    }

                    // Pattern A: 1 - 2·sin²(t) → cos(2t)
                    //   Requires: is_pos_one=true, trig_is_sin=true, coeff=-2
                    // Pattern B: 2·cos²(t) - 1 → cos(2t)
                    //   Requires: is_neg_one=true, trig_is_sin=false, coeff=+2
                    // Pattern C: -1 + 2·cos²(t) → cos(2t)
                    //   Requires: is_neg_one=true, trig_is_sin=false, coeff=+2
                    // Pattern D: -2·sin²(t) + 1 → cos(2t)
                    //   Same as A with different ordering

                    let matches = (is_pos_one && trig_is_sin && coeff == neg_two_rat)
                        || (is_neg_one && !trig_is_sin && coeff == two_rat);

                    if !matches {
                        continue;
                    }

                    // Build cos(2t)
                    let two = ctx.num(2);
                    let double_arg = ctx.add(Expr::Mul(two, trig_arg));
                    let cos_2t = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![double_arg]);

                    let desc = if trig_is_sin {
                        "1 - 2·sin²(t) = cos(2t)"
                    } else {
                        "2·cos²(t) - 1 = cos(2t)"
                    };

                    return Some(Rewrite::new(cos_2t).desc(desc));
                }
            }
        }

        None
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // POST only: prevents oscillation with HalfAngleExpansion and other trig
        // rules during TRANSFORM. In POST no expansion rules fire, so contraction
        // is stable and normalises 2cos²(t)-1 / 1-2sin²(t) to cos(2t).
        crate::phase::PhaseMask::POST
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB))
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200
    }
}

impl Cos2xAdditiveContractionRule {
    /// Extract (trig_arg, is_sin, coefficient) from a term like ±k·sin²(t) or ±k·cos²(t).
    /// Returns the argument to the trig function, whether it's sin (true) or cos (false),
    /// and the signed coefficient (including the sign from Neg).
    fn extract_coeff_trig_squared(
        ctx: &cas_ast::Context,
        term: ExprId,
    ) -> Option<(ExprId, bool, num_rational::BigRational)> {
        let two_rat = num_rational::BigRational::from_integer(2.into());

        // Handle negation: Neg(inner) → extract from inner with negated coeff
        let (base_term, sign) = if let Expr::Neg(inner) = ctx.get(term) {
            (*inner, num_rational::BigRational::from_integer((-1).into()))
        } else {
            (term, num_rational::BigRational::from_integer(1.into()))
        };

        // Flatten multiplication factors
        let mut factors = Vec::new();
        let mut stack = vec![base_term];
        while let Some(curr) = stack.pop() {
            if let Expr::Mul(l, r) = ctx.get(curr) {
                stack.push(*l);
                stack.push(*r);
            } else {
                factors.push(curr);
            }
        }

        // Find trig²(t) factor and numeric coefficient
        let mut trig_arg = None;
        let mut is_sin = false;
        let mut trig_idx = None;
        let mut numeric_coeff = sign;

        for (i, &f) in factors.iter().enumerate() {
            // Check for trig²(t) = Pow(trig(t), 2)
            if let Expr::Pow(base, exp) = ctx.get(f) {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if *n == two_rat {
                        if let Expr::Function(fn_id, args) = ctx.get(*base) {
                            if args.len() == 1 {
                                let builtin = ctx.builtin_of(*fn_id);
                                if matches!(builtin, Some(cas_ast::BuiltinFn::Sin)) {
                                    trig_arg = Some(args[0]);
                                    is_sin = true;
                                    trig_idx = Some(i);
                                    break;
                                } else if matches!(builtin, Some(cas_ast::BuiltinFn::Cos)) {
                                    trig_arg = Some(args[0]);
                                    is_sin = false;
                                    trig_idx = Some(i);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        let trig_arg = trig_arg?;
        let trig_idx = trig_idx?;

        // Multiply remaining factors to get the coefficient
        for (i, &f) in factors.iter().enumerate() {
            if i == trig_idx {
                continue;
            }
            if let Expr::Number(n) = ctx.get(f) {
                numeric_coeff *= n.clone();
            } else {
                // Non-numeric factor: this isn't a simple k·trig²(t) pattern
                return None;
            }
        }

        Some((trig_arg, is_sin, numeric_coeff))
    }
}

// =============================================================================
// AngleSumFractionToTanRule
// (sin(a)cos(b) + cos(a)sin(b)) / (cos(a)cos(b) - sin(a)sin(b)) → tan(a+b)
// (sin(a)cos(b) - cos(a)sin(b)) / (cos(a)cos(b) + sin(a)sin(b)) → tan(a-b)
// =============================================================================
// This rule contracts expanded angle-addition fractions back to tan.
// It targets Div nodes only, so it cannot loop with AngleIdentityRule
// (which targets Function nodes with sin/cos of Add/Sub arguments).
//
// Also handles the case where the numerator/denominator has extra common
// factors (e.g., multiplied through by cos²), by first trying the
// bare 2-term pattern.

pub struct AngleSumFractionToTanRule;

impl crate::rule::Rule for AngleSumFractionToTanRule {
    fn name(&self) -> &str {
        "Angle Sum Fraction to Tan"
    }

    fn priority(&self) -> i32 {
        190 // Below contraction rules at 200, above normal
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only match Div nodes
        let (num, den) = as_div(ctx, expr)?;

        // Try angle-sum pattern: numerator = sin(a)cos(b) + cos(a)sin(b)
        //                        denominator = cos(a)cos(b) - sin(a)sin(b)
        if let Some((a, b)) = self.try_match_sum(ctx, num, den) {
            let sum_arg = ctx.add(Expr::Add(a, b));
            let tan_result = ctx.call_builtin(BuiltinFn::Tan, vec![sum_arg]);
            return Some(
                Rewrite::new(tan_result)
                    .desc("(sin(a)cos(b)+cos(a)sin(b))/(cos(a)cos(b)-sin(a)sin(b)) = tan(a+b)"),
            );
        }

        // Try angle-difference pattern: numerator = sin(a)cos(b) - cos(a)sin(b)
        //                               denominator = cos(a)cos(b) + sin(a)sin(b)
        if let Some((a, b)) = self.try_match_diff(ctx, num, den) {
            let diff_arg = ctx.add(Expr::Sub(a, b));
            let tan_result = ctx.call_builtin(BuiltinFn::Tan, vec![diff_arg]);
            return Some(
                Rewrite::new(tan_result)
                    .desc("(sin(a)cos(b)-cos(a)sin(b))/(cos(a)cos(b)+sin(a)sin(b)) = tan(a-b)"),
            );
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

impl AngleSumFractionToTanRule {
    /// Semantic subtraction: matches Sub(a, b) OR Add(a, Neg(b)) OR Add(Neg(b), a).
    /// Returns (positive_part, subtracted_part).
    fn semantic_sub(&self, ctx: &cas_ast::Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
        // Direct Sub
        if let Some((l, r)) = as_sub(ctx, expr) {
            return Some((l, r));
        }
        // Add(a, Neg(b)) or Add(Neg(b), a)
        if let Expr::Add(l, r) = ctx.get(expr) {
            if let Some(inner) = as_neg(ctx, *r) {
                return Some((*l, inner));
            }
            if let Some(inner) = as_neg(ctx, *l) {
                return Some((*r, inner));
            }
        }
        None
    }

    /// Try to match numerator = sin(a)cos(b) + cos(a)sin(b) and
    ///                 denominator = cos(a)cos(b) - sin(a)sin(b)
    /// Returns Some((a, b)) if matched.
    fn try_match_sum(
        &self,
        ctx: &cas_ast::Context,
        num: ExprId,
        den: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // Numerator: A + B where A = sin(a)cos(b), B = cos(a)sin(b)
        let (nl, nr) = crate::helpers::as_add(ctx, num)?;
        let (a, b) = self.extract_sin_cos_product_pair(ctx, nl, nr)?;

        // Denominator: C - D where C = cos(a)cos(b), D = sin(a)sin(b)
        // Handles both Sub(C, D) and Add(C, Neg(D))
        let (dl, dr) = self.semantic_sub(ctx, den)?;
        let (a2, b2) = self.extract_cos_cos_minus_sin_sin(ctx, dl, dr)?;

        // Verify arguments match
        if self.args_equal(ctx, a, a2) && self.args_equal(ctx, b, b2) {
            return Some((a, b));
        }
        // Try (a,b) swapped in denominator
        if self.args_equal(ctx, a, b2) && self.args_equal(ctx, b, a2) {
            return Some((a, b));
        }

        None
    }

    /// Try to match numerator = sin(a)cos(b) - cos(a)sin(b) and
    ///                 denominator = cos(a)cos(b) + sin(a)sin(b)
    /// Returns Some((a, b)) if matched.
    fn try_match_diff(
        &self,
        ctx: &cas_ast::Context,
        num: ExprId,
        den: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // Numerator: A - B where A = sin(a)cos(b), B = cos(a)sin(b)
        // Handles both Sub(A, B) and Add(A, Neg(B))
        let (nl, nr) = self.semantic_sub(ctx, num)?;
        let (a, b) = self.extract_sin_cos_product_pair(ctx, nl, nr)?;

        // Denominator: C + D where C = cos(a)cos(b), D = sin(a)sin(b)
        let (dl, dr) = crate::helpers::as_add(ctx, den)?;
        let (a2, b2) = self.extract_cos_cos_and_sin_sin(ctx, dl, dr)?;

        // Verify arguments match
        if self.args_equal(ctx, a, a2) && self.args_equal(ctx, b, b2) {
            return Some((a, b));
        }
        if self.args_equal(ctx, a, b2) && self.args_equal(ctx, b, a2) {
            return Some((a, b));
        }

        None
    }

    /// Extract (a, b) from a pair where one is sin(a)·cos(b) and the other is cos(a)·sin(b).
    /// The first term should be the sin(a)·cos(b) term (matches the sign convention).
    fn extract_sin_cos_product_pair(
        &self,
        ctx: &cas_ast::Context,
        term1: ExprId,
        term2: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // Try: term1 = sin(a)·cos(b), term2 = cos(a)·sin(b)
        if let Some((sin_arg1, cos_arg1)) = self.extract_sin_times_cos(ctx, term1) {
            if let Some((sin_arg2, cos_arg2)) = self.extract_sin_times_cos(ctx, term2) {
                // sin(a)·cos(b) paired with cos(a)·sin(b) means:
                // sin_arg1=a, cos_arg1=b, cos_arg2=a, sin_arg2=b
                if self.args_equal(ctx, sin_arg1, cos_arg2)
                    && self.args_equal(ctx, cos_arg1, sin_arg2)
                {
                    return Some((sin_arg1, cos_arg1));
                }
            }
        }

        // Also try with mul_leaves for flattened products
        if let Some(result) = self.try_extract_via_mul_leaves(ctx, term1, term2) {
            return Some(result);
        }

        None
    }

    /// Extract (sin_arg, cos_arg) from sin(x)·cos(y) product.
    fn extract_sin_times_cos(
        &self,
        ctx: &cas_ast::Context,
        expr: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        let (l, r) = as_mul(ctx, expr)?;

        // Try l=sin, r=cos
        if let Some(sin_arg) = extract_trig_arg_named(ctx, l, BuiltinFn::Sin.name()) {
            if let Some(cos_arg) = extract_trig_arg_named(ctx, r, BuiltinFn::Cos.name()) {
                return Some((sin_arg, cos_arg));
            }
        }
        // Try l=cos, r=sin
        if let Some(cos_arg) = extract_trig_arg_named(ctx, l, BuiltinFn::Cos.name()) {
            if let Some(sin_arg) = extract_trig_arg_named(ctx, r, BuiltinFn::Sin.name()) {
                return Some((sin_arg, cos_arg));
            }
        }

        None
    }

    /// Try extracting the pattern from flattened mul_leaves.
    /// Handles cases like Mul(Mul(sin(a), cos(b)), Mul(cos(a), sin(b)))
    /// where mul_leaves gives [sin(a), cos(b)] and [cos(a), sin(b)].
    fn try_extract_via_mul_leaves(
        &self,
        ctx: &cas_ast::Context,
        term1: ExprId,
        term2: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        let factors1 = crate::nary::mul_leaves(ctx, term1);
        let factors2 = crate::nary::mul_leaves(ctx, term2);

        // Each term should have exactly 2 trig factors
        if factors1.len() != 2 || factors2.len() != 2 {
            return None;
        }

        // Try all combinations to find (sin(a), cos(b)) and (cos(a), sin(b))
        let sin1 = self.find_trig_in_factors(ctx, &factors1, BuiltinFn::Sin);
        let cos1 = self.find_trig_in_factors(ctx, &factors1, BuiltinFn::Cos);
        let sin2 = self.find_trig_in_factors(ctx, &factors2, BuiltinFn::Sin);
        let cos2 = self.find_trig_in_factors(ctx, &factors2, BuiltinFn::Cos);

        if let (Some(sin_arg1), Some(cos_arg1), Some(sin_arg2), Some(cos_arg2)) =
            (sin1, cos1, sin2, cos2)
        {
            // Pattern: sin(a)·cos(b) and cos(a)·sin(b)
            // sin_arg1=a, cos_arg1=b, cos_arg2=a, sin_arg2=b
            if self.args_equal(ctx, sin_arg1, cos_arg2) && self.args_equal(ctx, cos_arg1, sin_arg2)
            {
                return Some((sin_arg1, cos_arg1));
            }
        }

        None
    }

    /// Find a trig function argument in a list of factors.
    fn find_trig_in_factors(
        &self,
        ctx: &cas_ast::Context,
        factors: &[ExprId],
        target: BuiltinFn,
    ) -> Option<ExprId> {
        for &f in factors {
            if let Some(arg) = extract_trig_arg_named(ctx, f, target.name()) {
                return Some(arg);
            }
        }
        None
    }

    /// Extract (a, b) from cos(a)·cos(b) minus sin(a)·sin(b).
    fn extract_cos_cos_minus_sin_sin(
        &self,
        ctx: &cas_ast::Context,
        left: ExprId,
        right: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // left = cos(a)·cos(b)
        let (cos_a, cos_b) = self.extract_same_trig_product(ctx, left, BuiltinFn::Cos)?;
        // right = sin(a)·sin(b)
        let (sin_a, sin_b) = self.extract_same_trig_product(ctx, right, BuiltinFn::Sin)?;

        // Verify argument consistency: cos_a matches sin_a, cos_b matches sin_b
        if self.args_equal(ctx, cos_a, sin_a) && self.args_equal(ctx, cos_b, sin_b) {
            return Some((cos_a, cos_b));
        }
        // Or: cos_a matches sin_b, cos_b matches sin_a
        if self.args_equal(ctx, cos_a, sin_b) && self.args_equal(ctx, cos_b, sin_a) {
            return Some((cos_a, cos_b));
        }

        None
    }

    /// Extract (a, b) from cos(a)·cos(b) and sin(a)·sin(b) (both positive, for sum pattern).
    fn extract_cos_cos_and_sin_sin(
        &self,
        ctx: &cas_ast::Context,
        left: ExprId,
        right: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // Try left=cos·cos, right=sin·sin
        if let Some((cos_a, cos_b)) = self.extract_same_trig_product(ctx, left, BuiltinFn::Cos) {
            if let Some((sin_a, sin_b)) = self.extract_same_trig_product(ctx, right, BuiltinFn::Sin)
            {
                if self.args_equal(ctx, cos_a, sin_a) && self.args_equal(ctx, cos_b, sin_b) {
                    return Some((cos_a, cos_b));
                }
                if self.args_equal(ctx, cos_a, sin_b) && self.args_equal(ctx, cos_b, sin_a) {
                    return Some((cos_a, cos_b));
                }
            }
        }
        // Try left=sin·sin, right=cos·cos
        if let Some((sin_a, sin_b)) = self.extract_same_trig_product(ctx, left, BuiltinFn::Sin) {
            if let Some((cos_a, cos_b)) = self.extract_same_trig_product(ctx, right, BuiltinFn::Cos)
            {
                if self.args_equal(ctx, cos_a, sin_a) && self.args_equal(ctx, cos_b, sin_b) {
                    return Some((cos_a, cos_b));
                }
                if self.args_equal(ctx, cos_a, sin_b) && self.args_equal(ctx, cos_b, sin_a) {
                    return Some((cos_a, cos_b));
                }
            }
        }

        None
    }

    /// Extract two arguments from f(a)·f(b) where f is a specific trig function.
    fn extract_same_trig_product(
        &self,
        ctx: &cas_ast::Context,
        expr: ExprId,
        target: BuiltinFn,
    ) -> Option<(ExprId, ExprId)> {
        // Direct Mul
        if let Some((l, r)) = as_mul(ctx, expr) {
            if let Some(a) = extract_trig_arg_named(ctx, l, target.name()) {
                if let Some(b) = extract_trig_arg_named(ctx, r, target.name()) {
                    return Some((a, b));
                }
            }
        }

        // Flattened mul_leaves
        let factors = crate::nary::mul_leaves(ctx, expr);
        if factors.len() == 2 {
            if let Some(a) = extract_trig_arg_named(ctx, factors[0], target.name()) {
                if let Some(b) = extract_trig_arg_named(ctx, factors[1], target.name()) {
                    return Some((a, b));
                }
            }
        }

        None
    }

    fn args_equal(&self, ctx: &cas_ast::Context, a: ExprId, b: ExprId) -> bool {
        a == b || crate::ordering::compare_expr(ctx, a, b) == std::cmp::Ordering::Equal
    }
}
