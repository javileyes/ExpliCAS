//! Polynomial GCD structural rule.
//!
//! Implements `poly_gcd(a, b)` which finds the structural GCD of two expressions
//! by collecting multiplicative factors and intersecting them.
//!
//! Example:
//! ```text
//! poly_gcd((1+x)^3 * (2+y), (1+x)^2 * (3+z)) = (1+x)^2
//! poly_gcd(a*g, b*g) = g
//! ```
//!
//! This allows Mathematica/Symbolica-style polynomial GCD without expanding.

use crate::engine::Simplifier;
use crate::options::EvalOptions;
use crate::phase::PhaseMask;
use crate::rule::{Rewrite, Rule};
use cas_ast::{Context, ExprId};
use cas_formatter::DisplayExpr;
use cas_math::gcd_zippel_modp::ZippelPreset;
use cas_math::poly_gcd_dispatch::{
    classify_pre_evaluate_for_gcd, compute_poly_gcd_unified_with, GcdPreEvalDirective,
};
use cas_math::poly_gcd_mode::{try_parse_poly_gcd_call, GcdGoal, GcdMode};

#[cfg(test)]
use cas_ast::Expr;

/// Pre-evaluate an expression to resolve specific function wrappers.
///
/// SAFETY:
/// - Only evaluates `expand()` (explicit user intent) and `__hold` (internal wrapper)
/// - Uses StepsMode::Off and ExpandPolicy::Never to avoid recursive work
/// - Does NOT evaluate `factor()` or `simplify()` (too expensive for GCD path)
/// - Avoids recursion: won't trigger poly_gcd from within poly_gcd
fn pre_evaluate_for_gcd(ctx: &mut Context, expr: ExprId) -> ExprId {
    use crate::options::StepsMode;
    use crate::phase::ExpandPolicy;

    match classify_pre_evaluate_for_gcd(ctx, expr) {
        GcdPreEvalDirective::EvaluateExpand { expand_call } => {
            // expand() is explicitly requested by user - evaluate it
            let opts = EvalOptions {
                steps_mode: StepsMode::Off,
                shared: crate::phase::SharedSemanticConfig {
                    expand_policy: ExpandPolicy::Off,
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut simplifier = Simplifier::with_profile(&opts);
            simplifier.set_steps_mode(StepsMode::Off);

            std::mem::swap(&mut simplifier.context, ctx);
            let (result, _) = simplifier.expand(expand_call);
            std::mem::swap(&mut simplifier.context, ctx);
            result
        }
        GcdPreEvalDirective::UnwrapHold { inner } => inner,
        GcdPreEvalDirective::Keep => expr,
    }
}

/// Fraction-cancellation helper forwarded to `cas_math`.
pub fn gcd_shallow_for_fraction(ctx: &mut Context, num: ExprId, den: ExprId) -> (ExprId, String) {
    cas_math::poly_gcd_structural::gcd_shallow_for_fraction(ctx, num, den)
}

// =============================================================================
// Unified GCD dispatcher
// =============================================================================

/// Compute GCD using specified mode, returning (result, description).
///
/// The `goal` parameter determines allowed methods:
/// - `UserPolyGcd`: Full pipeline (Structural → Exact → Modp)
/// - `CancelFraction`: Safe methods only (Structural → Exact), modp BLOCKED
pub fn compute_poly_gcd_unified(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    goal: GcdGoal,
    mode: GcdMode,
    modp_preset: Option<ZippelPreset>,
    modp_main_var: Option<usize>,
) -> (ExprId, String) {
    compute_poly_gcd_unified_with(
        ctx,
        a,
        b,
        goal,
        mode,
        modp_preset,
        modp_main_var,
        pre_evaluate_for_gcd,
        |render_ctx, id| {
            format!(
                "{}",
                DisplayExpr {
                    context: render_ctx,
                    id
                }
            )
        },
    )
}

// =============================================================================
// REPL function rule
// =============================================================================

/// Rule for poly_gcd(a, b) function.
/// Computes structural GCD of two polynomial expressions.
pub struct PolyGcdRule;

impl Rule for PolyGcdRule {
    fn name(&self) -> &str {
        "Polynomial GCD"
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::CORE | PhaseMask::TRANSFORM
    }

    fn priority(&self) -> i32 {
        200 // High priority to evaluate early
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let parsed = try_parse_poly_gcd_call(ctx, expr)?;
        let (result, description) = compute_poly_gcd_unified(
            ctx,
            parsed.lhs,
            parsed.rhs,
            GcdGoal::UserPolyGcd,
            parsed.mode,
            parsed.modp_preset,
            parsed.modp_main_var,
        );

        // Wrap result in __hold() to prevent further simplification
        let held_gcd = cas_ast::hold::wrap_hold(ctx, result);
        Some(Rewrite::simple(held_gcd, description))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cas_math::poly_gcd_structural::poly_gcd_structural;
    use num_bigint::BigInt;

    fn setup_ctx() -> Context {
        Context::new()
    }

    #[test]
    fn test_poly_gcd_simple_common_factor() {
        let mut ctx = setup_ctx();

        // x+1
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_1 = ctx.add(Expr::Add(x, one));

        // y+2
        let y = ctx.var("y");
        let two = ctx.num(2);
        let y_plus_2 = ctx.add(Expr::Add(y, two));

        // (x+1) * (y+2)
        let a = ctx.add(Expr::Mul(x_plus_1, y_plus_2));

        // z+3
        let z = ctx.var("z");
        let three = ctx.num(3);
        let z_plus_3 = ctx.add(Expr::Add(z, three));

        // (x+1) * (z+3)
        let b = ctx.add(Expr::Mul(x_plus_1, z_plus_3));

        // GCD should be (x+1)
        let gcd = poly_gcd_structural(&mut ctx, a, b);

        // Verify it's x+1
        assert_eq!(gcd, x_plus_1);
    }

    #[test]
    fn test_poly_gcd_with_powers() {
        let mut ctx = setup_ctx();

        // x+1
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_1 = ctx.add(Expr::Add(x, one));

        // (x+1)^3
        let three = ctx.num(3);
        let pow3 = ctx.add(Expr::Pow(x_plus_1, three));

        // (x+1)^2
        let two = ctx.num(2);
        let pow2 = ctx.add(Expr::Pow(x_plus_1, two));

        // GCD((x+1)^3, (x+1)^2) = (x+1)^2
        let gcd = poly_gcd_structural(&mut ctx, pow3, pow2);

        // Should be (x+1)^2
        let gcd_str = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: gcd
            }
        );
        assert!(gcd_str.contains("x") && gcd_str.contains("1"));
    }

    #[test]
    fn test_poly_gcd_no_common() {
        let mut ctx = setup_ctx();

        // x
        let x = ctx.var("x");
        // y
        let y = ctx.var("y");

        // GCD(x, y) = 1 (no structural common factor)
        let gcd = poly_gcd_structural(&mut ctx, x, y);

        // Should be 1
        if let Expr::Number(n) = ctx.get(gcd) {
            assert_eq!(*n, num_rational::BigRational::from_integer(BigInt::from(1)));
        } else {
            panic!("Expected number 1");
        }
    }
}
