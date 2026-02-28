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
use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_math::gcd_zippel_modp::ZippelPreset;
use cas_math::poly_gcd_dispatch::compute_poly_gcd_unified_with;
use cas_math::poly_gcd_mode::parse_modp_options;

use cas_math::poly_gcd_mode::{parse_gcd_mode, GcdGoal, GcdMode};

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

    // Only process specific wrappers that need evaluation
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let fn_id = *fn_id;
        let args = args.clone();

        // Check for expand() via builtin
        if matches!(ctx.builtin_of(fn_id), Some(cas_ast::BuiltinFn::Expand)) {
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

            // Transfer context
            std::mem::swap(&mut simplifier.context, ctx);
            let (result, _) = simplifier.expand(expr); // Use expand() specifically
                                                       // Transfer back
            std::mem::swap(&mut simplifier.context, ctx);
            return result;
        }

        // __hold is an internal wrapper - unwrap it using canonical helper
        if ctx.is_builtin(fn_id, cas_ast::BuiltinFn::Hold) && !args.is_empty() {
            return args[0]; // Just unwrap, don't recurse
        }

        // factor() and simplify() are TOO EXPENSIVE for GCD path
        // Leave them as-is and let the converter handle or fail gracefully
        let name = ctx.sym_name(fn_id);
        if name == "factor" || name == "simplify" {
            // Don't pre-evaluate these - they could be O(expensive)
            // The GCD will fall back to structural if conversion fails
            return expr;
        }
    }
    expr
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
        let (fn_id, args) = if let Expr::Function(fn_id, args) = ctx.get(expr) {
            (*fn_id, args.clone())
        } else {
            return None;
        };
        {
            let name = ctx.sym_name(fn_id);
            // Match poly_gcd, pgcd with 2-4 arguments
            let is_poly_gcd = name == "poly_gcd" || name == "pgcd";

            if is_poly_gcd && args.len() >= 2 && args.len() <= 4 {
                let a = args[0];
                let b = args[1];

                // Parse mode from 3rd argument (or default to Structural)
                let mode = if args.len() >= 3 {
                    parse_gcd_mode(ctx, args[2])
                } else {
                    GcdMode::Structural
                };

                // Parse modp options from remaining args
                let (modp_preset, modp_main_var) = if args.len() >= 4 {
                    parse_modp_options(ctx, &args[3..])
                } else if args.len() == 3 && mode == GcdMode::Modp {
                    // No extra args for modp, use defaults
                    (None, None)
                } else {
                    (None, None)
                };

                let (result, description) = compute_poly_gcd_unified(
                    ctx,
                    a,
                    b,
                    GcdGoal::UserPolyGcd,
                    mode,
                    modp_preset,
                    modp_main_var,
                );

                // Wrap result in __hold() to prevent further simplification
                let held_gcd = cas_ast::hold::wrap_hold(ctx, result);

                return Some(Rewrite::simple(held_gcd, description));
            }
        }

        None
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
