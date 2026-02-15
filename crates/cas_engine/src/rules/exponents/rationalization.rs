use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Zero};

// =============================================================================
// RationalizeLinearSqrtDenRule: 1/(sqrt(t)+c) → (sqrt(t)-c)/(t-c²)
// =============================================================================
// Rationalizes denominators with linear sqrt terms by multiplying by conjugate.
// This is a canonical transformation that eliminates radicals from denominators.
//
// Examples:
//   1/(sqrt(2)+1) → (sqrt(2)-1)/1 = sqrt(2)-1
//   1/(sqrt(3)+1) → (sqrt(3)-1)/2
//   1/(sqrt(u)+1) → (sqrt(u)-1)/(u-1)
//   2/(sqrt(3)-1) → 2*(sqrt(3)+1)/2 = sqrt(3)+1
//
// Guard: Only apply when result is simpler (no radicals in denominator)
// =============================================================================
define_rule!(
    RationalizeLinearSqrtDenRule,
    "Rationalize Linear Sqrt Denominator",
    |ctx, expr| {
        // Match Div(num, Add(sqrt_term, const_term)) or Div(num, Sub(...))
        let (numerator, denominator) = match ctx.get(expr) {
            Expr::Div(n, d) => (*n, *d),
            _ => return None,
        };

        // Parse denominator as sqrt(t) ± c
        let (sqrt_arg, const_part, is_plus) = match ctx.get(denominator) {
            Expr::Add(l, r) => {
                let (l, r) = (*l, *r);
                // sqrt(t) + c or c + sqrt(t)
                if let Some(arg) = extract_sqrt_arg(ctx, l) {
                    // sqrt(t) + c
                    (arg, r, true)
                } else if let Some(arg) = extract_sqrt_arg(ctx, r) {
                    // c + sqrt(t) - treat as sqrt(t) + c
                    (arg, l, true)
                } else {
                    return None;
                }
            }
            Expr::Sub(l, r) => {
                let (l, r) = (*l, *r);
                // sqrt(t) - c
                if let Some(arg) = extract_sqrt_arg(ctx, l) {
                    (arg, r, false)
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // const_part should be a simple expression (number or variable)
        // For now, focus on integer constants where we can verify simplification
        let const_is_simple = matches!(ctx.get(const_part), Expr::Number(_) | Expr::Variable(_));
        if !const_is_simple {
            // Skip complex const_part to avoid explosion
            return None;
        }

        // Build conjugate: if denom = sqrt(t)+c, conjugate = sqrt(t)-c
        // Result: num*(sqrt(t)-c) / ((sqrt(t)+c)*(sqrt(t)-c)) = num*(sqrt(t)-c) / (t - c²)
        let half_exp = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let sqrt_t = ctx.add(Expr::Pow(sqrt_arg, half_exp));

        let conjugate_num = if is_plus {
            // denom = sqrt(t)+c → conjugate = sqrt(t)-c
            ctx.add(Expr::Sub(sqrt_t, const_part))
        } else {
            // denom = sqrt(t)-c → conjugate = sqrt(t)+c
            ctx.add(Expr::Add(sqrt_t, const_part))
        };

        // New numerator: original_num * conjugate
        let new_num = ctx.add(Expr::Mul(numerator, conjugate_num));

        // New denominator: t - c²
        let two = ctx.num(2);
        let c_squared = ctx.add(Expr::Pow(const_part, two));
        let new_den = ctx.add(Expr::Sub(sqrt_arg, c_squared));

        let result = ctx.add(Expr::Div(new_num, new_den));

        Some(Rewrite::new(result).desc("Rationalize: multiply by conjugate"))
    }
);

/// Extract the argument from a sqrt expression (Pow(t, 1/2))
pub(super) fn extract_sqrt_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            let half = num_rational::BigRational::new(1.into(), 2.into());
            if *n == half {
                return Some(*base);
            }
        }
    }
    None
}

// =============================================================================
// RationalizeSumOfSqrtsDenRule: k/(sqrt(p)+sqrt(q)) → k*(sqrt(p)-sqrt(q))/(p-q)
// =============================================================================
// Rationalizes denominators with sum of two square roots.
//
// Examples:
//   3/(sqrt(2)+sqrt(3)) → 3*(sqrt(2)-sqrt(3))/(2-3) = -3*(sqrt(2)-sqrt(3))
//   1/(sqrt(5)+sqrt(2)) → (sqrt(5)-sqrt(2))/3
// =============================================================================
define_rule!(
    RationalizeSumOfSqrtsDenRule,
    "Rationalize Sum of Sqrts Denominator",
    |ctx, expr| {
        // Match Div(num, Add/Sub of two sqrts)
        let (numerator, denominator) = match ctx.get(expr) {
            Expr::Div(n, d) => (*n, *d),
            _ => return None,
        };

        // Parse denominator as sqrt(p) ± sqrt(q)
        let (sqrt_p_arg, sqrt_q_arg, is_plus) = match ctx.get(denominator) {
            Expr::Add(l, r) => {
                let (l, r) = (*l, *r);
                // sqrt(p) + sqrt(q)
                let p_arg = extract_sqrt_arg(ctx, l)?;
                let q_arg = extract_sqrt_arg(ctx, r)?;
                (p_arg, q_arg, true)
            }
            Expr::Sub(l, r) => {
                let (l, r) = (*l, *r);
                // sqrt(p) - sqrt(q)
                let p_arg = extract_sqrt_arg(ctx, l)?;
                let q_arg = extract_sqrt_arg(ctx, r)?;
                (p_arg, q_arg, false)
            }
            _ => return None,
        };

        // Build conjugate multiplication
        // If denom = sqrt(p)+sqrt(q), conjugate is sqrt(p)-sqrt(q)
        // New denominator = (sqrt(p)+sqrt(q))*(sqrt(p)-sqrt(q)) = p - q
        let half_exp = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let sqrt_p = ctx.add(Expr::Pow(sqrt_p_arg, half_exp));
        let half_exp2 = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let sqrt_q = ctx.add(Expr::Pow(sqrt_q_arg, half_exp2));

        let conjugate = if is_plus {
            // denom = sqrt(p)+sqrt(q) → conjugate = sqrt(p)-sqrt(q)
            ctx.add(Expr::Sub(sqrt_p, sqrt_q))
        } else {
            // denom = sqrt(p)-sqrt(q) → conjugate = sqrt(p)+sqrt(q)
            ctx.add(Expr::Add(sqrt_p, sqrt_q))
        };

        // New numerator: original_num * conjugate
        let new_num = ctx.add(Expr::Mul(numerator, conjugate));

        // New denominator: p - q
        let new_den = ctx.add(Expr::Sub(sqrt_p_arg, sqrt_q_arg));

        let result = ctx.add(Expr::Div(new_num, new_den));

        Some(Rewrite::new(result).desc("Rationalize: (sqrt(p)±sqrt(q)) multiply by conjugate"))
    }
);

// =============================================================================
// CubeRootDenRationalizeRule: k/(1+u^(1/3)) → k*(1-u^(1/3)+u^(2/3))/(1+u)
// =============================================================================
// Uses the sum of cubes identity: 1 + r³ = (1 + r)(1 - r + r²)
// So: 1/(1+r) = (1-r+r²)/(1+r³)
// With r = u^(1/3), r³ = u
//
// Similarly for difference: 1 - r³ = (1 - r)(1 + r + r²)
// So: 1/(1-r) = (1+r+r²)/(1-r³)
//
// Examples:
//   1/(1+u^(1/3)) → (1-u^(1/3)+u^(2/3))/(1+u)
//   1/(1-u^(1/3)) → (1+u^(1/3)+u^(2/3))/(1-u)
// =============================================================================
define_rule!(
    CubeRootDenRationalizeRule,
    "Rationalize Cube Root Denominator",
    |ctx, expr| {
        // Match Div(num, Add/Sub with 1 and cube root)
        let (numerator, denominator) = match ctx.get(expr) {
            Expr::Div(n, d) => (*n, *d),
            _ => return None,
        };

        // Parse denominator as 1 ± u^(1/3)
        let (cbrt_base, is_plus) = match ctx.get(denominator) {
            Expr::Add(l, r) => {
                let (l, r) = (*l, *r);
                // 1 + r or r + 1
                if is_one(ctx, l) {
                    if let Some(base) = extract_cbrt_arg(ctx, r) {
                        (base, true)
                    } else {
                        return None;
                    }
                } else if is_one(ctx, r) {
                    if let Some(base) = extract_cbrt_arg(ctx, l) {
                        (base, true)
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            Expr::Sub(l, r) => {
                let (l, r) = (*l, *r);
                // 1 - r
                if is_one(ctx, l) {
                    if let Some(base) = extract_cbrt_arg(ctx, r) {
                        (base, false) // 1 - r
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Build the rationalization factor
        // For 1+r: factor is (1 - r + r²)
        // For 1-r: factor is (1 + r + r²)
        let one = ctx.num(1);
        let one_third = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            3.into(),
        )));
        let two_thirds = ctx.add(Expr::Number(num_rational::BigRational::new(
            2.into(),
            3.into(),
        )));

        let r = ctx.add(Expr::Pow(cbrt_base, one_third)); // u^(1/3)
        let r_squared = ctx.add(Expr::Pow(cbrt_base, two_thirds)); // u^(2/3)

        let factor = if is_plus {
            // 1 - r + r² = (1 - r) + r² = 1 + r² - r
            let one_minus_r = ctx.add(Expr::Sub(one, r));
            ctx.add(Expr::Add(one_minus_r, r_squared))
        } else {
            // 1 + r + r² = (1 + r) + r²
            let one_plus_r = ctx.add(Expr::Add(one, r));
            ctx.add(Expr::Add(one_plus_r, r_squared))
        };

        // New numerator: original_num * factor
        let new_num = ctx.add(Expr::Mul(numerator, factor));

        // New denominator: 1 ± u (since r³ = u)
        let one2 = ctx.num(1);
        let new_den = if is_plus {
            ctx.add(Expr::Add(one2, cbrt_base)) // 1 + u
        } else {
            ctx.add(Expr::Sub(one2, cbrt_base)) // 1 - u
        };

        let result = ctx.add(Expr::Div(new_num, new_den));

        Some(Rewrite::new(result).desc("Rationalize: cube root denominator via sum of cubes"))
    }
);

/// Check if expression is the number 1
fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

/// Extract the base from a cube root expression (Pow(t, 1/3))
fn extract_cbrt_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            let one_third = num_rational::BigRational::new(1.into(), 3.into());
            if *n == one_third {
                return Some(*base);
            }
        }
    }
    None
}

// =============================================================================
// RootMergeMulRule: sqrt(a) * sqrt(b) → sqrt(a*b)
// =============================================================================
// Merges products of square roots into a single root.
// This is valid for non-negative real a and b.
//
// Examples:
//   sqrt(u) * sqrt(b) → sqrt(u*b)
//   u^(1/2) * b^(1/2) → (u*b)^(1/2)
//
// Requires: a ≥ 0 and b ≥ 0 (or they are squared terms)
// =============================================================================
pub struct RootMergeMulRule;

impl crate::rule::Rule for RootMergeMulRule {
    fn name(&self) -> &str {
        "Merge Sqrt Product"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        use crate::domain::Proof;
        use crate::helpers::prove_nonnegative;

        // Match Mul(Pow(a, 1/2), Pow(b, 1/2))
        let (left, right) = match ctx.get(expr) {
            Expr::Mul(l, r) => (*l, *r),
            _ => return None,
        };

        // Both must be sqrt (Pow with exponent 1/2)
        let a = extract_sqrt_arg(ctx, left)?;
        let b = extract_sqrt_arg(ctx, right)?;

        // Check if both are provably non-negative
        let vd = parent_ctx.value_domain();
        let proof_a = prove_nonnegative(ctx, a, vd);
        let proof_b = prove_nonnegative(ctx, b, vd);

        // Only proceed if both are proven or we can safely assume
        // For educational mode, we apply with assumption
        let mode = parent_ctx.domain_mode();
        let can_apply = match mode {
            crate::domain::DomainMode::Generic => true, // Apply with assumption
            crate::domain::DomainMode::Assume => true,
            crate::domain::DomainMode::Strict => {
                proof_a == Proof::Proven && proof_b == Proof::Proven
            }
        };

        if !can_apply {
            return None;
        }

        // Build sqrt(a*b)
        let product = ctx.add(Expr::Mul(a, b));
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let result = ctx.add(Expr::Pow(product, half));

        let mut rewrite = Rewrite::new(result).desc("√a · √b = √(a·b)");

        // Add assumptions if needed
        if proof_a != Proof::Proven {
            rewrite = rewrite.assume(crate::assumptions::AssumptionEvent::nonnegative(ctx, a));
        }
        if proof_b != Proof::Proven {
            rewrite = rewrite.assume(crate::assumptions::AssumptionEvent::nonnegative(ctx, b));
        }

        Some(rewrite)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

// =============================================================================
// RootMergeDivRule: sqrt(a) / sqrt(b) → sqrt(a/b)
// =============================================================================
// Merges quotients of square roots into a single root.
// This is valid for non-negative real a and positive b.
//
// Examples:
//   sqrt(u) / sqrt(b) → sqrt(u/b)
//   u^(1/2) / b^(1/2) → (u/b)^(1/2)
//
// Requires: a ≥ 0 and b > 0
// =============================================================================
pub struct RootMergeDivRule;

impl crate::rule::Rule for RootMergeDivRule {
    fn name(&self) -> &str {
        "Merge Sqrt Quotient"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        use crate::domain::Proof;
        use crate::helpers::{prove_nonnegative, prove_positive};

        // Match Div(Pow(a, 1/2), Pow(b, 1/2))
        let (num, den) = match ctx.get(expr) {
            Expr::Div(n, d) => (*n, *d),
            _ => return None,
        };

        // Both must be sqrt (Pow with exponent 1/2)
        let a = extract_sqrt_arg(ctx, num)?;
        let b = extract_sqrt_arg(ctx, den)?;

        // Check domain conditions
        let vd = parent_ctx.value_domain();
        let proof_a = prove_nonnegative(ctx, a, vd);
        let proof_b = prove_positive(ctx, b, vd); // b > 0 for division

        let mode = parent_ctx.domain_mode();
        let can_apply = match mode {
            crate::domain::DomainMode::Generic => true,
            crate::domain::DomainMode::Assume => true,
            crate::domain::DomainMode::Strict => {
                proof_a == Proof::Proven && proof_b == Proof::Proven
            }
        };

        if !can_apply {
            return None;
        }

        // Build sqrt(a/b)
        let quotient = ctx.add(Expr::Div(a, b));
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let result = ctx.add(Expr::Pow(quotient, half));

        let mut rewrite = Rewrite::new(result).desc("√a / √b = √(a/b)");

        // Add assumptions if needed
        if proof_a != Proof::Proven {
            rewrite = rewrite.assume(crate::assumptions::AssumptionEvent::nonnegative(ctx, a));
        }
        if proof_b != Proof::Proven {
            rewrite = rewrite.assume(crate::assumptions::AssumptionEvent::positive(ctx, b));
        }

        Some(rewrite)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

// =============================================================================
// PowPowCancelReciprocalRule: (u^y)^(1/y) → u
// =============================================================================
// Cancels reciprocal exponents in nested powers.
// This is valid for u > 0 and y ≠ 0 in real domain.
//
// Examples:
//   (u^y)^(1/y) → u
//   (x^n)^(1/n) → x
//
// Requires: u > 0 (base), y ≠ 0 (exponent)
// =============================================================================
pub struct PowPowCancelReciprocalRule;

impl crate::rule::Rule for PowPowCancelReciprocalRule {
    fn name(&self) -> &str {
        "Cancel Reciprocal Exponents"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        use crate::domain::Proof;
        use crate::helpers::{prove_nonzero, prove_positive};

        // Match Pow(Pow(u, y), Div(1, y)) or Pow(Pow(u, y), exp) where exp = 1/y
        let (inner_pow, outer_exp) = match ctx.get(expr) {
            Expr::Pow(base, exp) => (*base, *exp),
            _ => return None,
        };

        // Inner must be Pow(u, y)
        let (u, y) = match ctx.get(inner_pow) {
            Expr::Pow(base, exp) => (*base, *exp),
            _ => return None,
        };

        // Outer exponent must be 1/y (either Div(1, y) or a number that equals 1/y)
        let is_reciprocal = match ctx.get(outer_exp) {
            Expr::Div(num, den) => {
                let (num, den) = (*num, *den);
                // Check if num is 1 and den equals y
                let is_one = matches!(ctx.get(num), Expr::Number(n) if n.is_one());
                let same_exp = den == y;
                is_one && same_exp
            }
            Expr::Number(outer_n) => {
                let outer_n = outer_n.clone();
                // Check if y is a number and outer_exp = 1/y
                if let Expr::Number(y_n) = ctx.get(y) {
                    !y_n.is_zero() && outer_n == y_n.recip()
                } else {
                    false
                }
            }
            _ => false,
        };

        if !is_reciprocal {
            return None;
        }

        // Check domain conditions
        let vd = parent_ctx.value_domain();
        let proof_u_pos = prove_positive(ctx, u, vd);
        let proof_y_nonzero = prove_nonzero(ctx, y);

        let mode = parent_ctx.domain_mode();
        let can_apply = match mode {
            crate::domain::DomainMode::Generic => true,
            crate::domain::DomainMode::Assume => true,
            crate::domain::DomainMode::Strict => {
                proof_u_pos == Proof::Proven && proof_y_nonzero == Proof::Proven
            }
        };

        if !can_apply {
            return None;
        }

        // Result is just u
        let mut rewrite = Rewrite::new(u).desc("(u^y)^(1/y) = u");

        // Add assumptions if needed
        if proof_u_pos != Proof::Proven {
            rewrite = rewrite.assume(crate::assumptions::AssumptionEvent::positive(ctx, u));
        }
        if proof_y_nonzero != Proof::Proven {
            rewrite = rewrite.assume(crate::assumptions::AssumptionEvent::nonzero(ctx, y));
        }

        Some(rewrite)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

// =============================================================================
// ReciprocalSqrtCanonRule: Canonicalize reciprocal sqrt forms to Pow(x, -1/2)
// =============================================================================
// Ensures all representations of "1/√x" converge to a single canonical AST:
//
//   Pattern 1: 1/√x      = Div(1, Pow(x, 1/2))     → Pow(x, -1/2)
//   Pattern 2: √x/x      = Div(Pow(x, 1/2), x)     → Pow(x, -1/2)
//   Pattern 3: √(x^(-1)) = Pow(Pow(x,-1), 1/2)     → already handled by PowerPowerRule
//
// GUARD: Only applied when the base contains symbols (variables).
// Pure numeric bases (e.g., 1/√2) are left as-is to avoid creating Pow(2, -1/2)
// forms that Strict-mode verification cannot fold back to √2/2.
//
// This is sound in RealOnly: all forms require x > 0, same definability domain.
// No cycle risk: NegativeExponentNormalizationRule only fires on INTEGER negative
// exponents, and -1/2 is not integer.
// =============================================================================

/// Check if an expression contains any symbolic (variable) nodes.
/// Returns false for pure numeric/constant expressions.
fn contains_symbol(ctx: &Context, e: ExprId) -> bool {
    match ctx.get(e) {
        Expr::Variable(_) => true,
        Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
        Expr::Neg(a) | Expr::Hold(a) => contains_symbol(ctx, *a),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            contains_symbol(ctx, *a) || contains_symbol(ctx, *b)
        }
        Expr::Function(_, args) => args.iter().any(|&a| contains_symbol(ctx, a)),
        Expr::Matrix { data, .. } => data.iter().any(|&a| contains_symbol(ctx, a)),
    }
}

pub struct ReciprocalSqrtCanonRule;

impl crate::rule::Rule for ReciprocalSqrtCanonRule {
    fn name(&self) -> &str {
        "Canonicalize Reciprocal Sqrt"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let (num, den) = match ctx.get(expr) {
            Expr::Div(n, d) => (*n, *d),
            _ => return None,
        };

        let neg_half = num_rational::BigRational::new((-1).into(), 2.into());

        // Pattern 1: Div(1, Pow(x, 1/2)) → Pow(x, -1/2)
        // i.e., 1/sqrt(x) → x^(-1/2)
        if let Expr::Number(n) = ctx.get(num) {
            if n.is_one() {
                if let Some(base) = extract_sqrt_arg(ctx, den) {
                    // Guard: skip pure numeric bases (e.g., 1/sqrt(2))
                    if !contains_symbol(ctx, base) {
                        return None;
                    }
                    let exp = ctx.add(Expr::Number(neg_half));
                    let result = ctx.add(Expr::Pow(base, exp));
                    return Some(crate::rule::Rewrite::new(result).desc("1/√x = x^(-1/2)"));
                }
            }
        }

        // Pattern 2: Div(Pow(x, 1/2), x) → Pow(x, -1/2)
        // i.e., sqrt(x)/x → x^(-1/2)
        if let Some(sqrt_base) = extract_sqrt_arg(ctx, num) {
            // Check if den == sqrt_base (structurally)
            if crate::ordering::compare_expr(ctx, sqrt_base, den) == std::cmp::Ordering::Equal {
                // Guard: skip pure numeric bases (e.g., sqrt(2)/2)
                if !contains_symbol(ctx, sqrt_base) {
                    return None;
                }
                let exp = ctx.add(Expr::Number(neg_half));
                let result = ctx.add(Expr::Pow(sqrt_base, exp));
                return Some(crate::rule::Rewrite::new(result).desc("√x/x = x^(-1/2)"));
            }
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Low
    }
}
