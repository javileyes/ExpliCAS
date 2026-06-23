//! Unified polynomial-GCD dispatcher shared across runtime crates.
//!
//! This module centralizes mode selection (`structural/exact/modp/auto`) and
//! soundness gates based on solve goals.

use crate::gcd_exact::{gcd_exact, normalize_multipoly, GcdExactBudget, GcdExactLayer};
use crate::gcd_zippel_modp::ZippelPreset;
use crate::multipoly::{multipoly_from_expr, PolyBudget};
use crate::poly_gcd_mode::{GcdGoal, GcdMode};
use crate::poly_gcd_structural::poly_gcd_structural;
use crate::poly_modp_conv::{compute_gcd_modp_expr_with_options, DEFAULT_PRIME};
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;

/// Verified modular-GCD fallback for genuinely multivariate inputs.
///
/// The exact Layer 1/2/2.5 pipeline (`gcd_exact`) returns `1` not only for
/// coprime inputs but ALSO when its heuristic seeds simply miss an existing
/// common factor — e.g. `gcd(x²−y², x²+2xy+y²)` returned `1` instead of `x+y`,
/// a soundness wrong-answer (falsely claiming coprimality). The Zippel modular
/// GCD is complete for these cases, but single-prime and therefore
/// probabilistic, so its candidate is accepted ONLY after EXACT polynomial
/// division confirms it divides BOTH inputs — guaranteeing we never return a
/// spurious factor. Returns `None` (caller keeps `1`) when the candidate is
/// constant or fails verification, so genuinely coprime inputs still resolve to
/// `1`.
fn try_verified_modp_multivar_gcd(ctx: &mut Context, a: ExprId, b: ExprId) -> Option<ExprId> {
    let candidate = compute_gcd_modp_expr_with_options(
        ctx,
        a,
        b,
        DEFAULT_PRIME,
        None,
        Some(ZippelPreset::Aggressive),
    )
    .ok()?;

    let budget = PolyBudget::default();
    let pg = multipoly_from_expr(ctx, candidate, &budget).ok()?;
    if pg.is_constant() {
        return None;
    }
    let pa = multipoly_from_expr(ctx, a, &budget).ok()?;
    let pb = multipoly_from_expr(ctx, b, &budget).ok()?;

    // `div_exact` requires identical variable sets, so align all three to the
    // shared (union) variable set before verifying exact divisibility.
    let mut vars: Vec<String> = pa
        .vars
        .iter()
        .chain(pb.vars.iter())
        .chain(pg.vars.iter())
        .cloned()
        .collect();
    vars.sort();
    vars.dedup();
    let pa = pa.align_vars(&vars);
    let pb = pb.align_vars(&vars);
    let pg_aligned = pg.align_vars(&vars);

    // EXACT verification: the candidate must divide both inputs with no
    // remainder. This is what makes the probabilistic modular GCD sound.
    pa.div_exact(&pg_aligned)?;
    pb.div_exact(&pg_aligned)?;

    Some(normalize_multipoly(ctx, &pg))
}

/// Classification of runtime pre-evaluation action for one GCD argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcdPreEvalDirective {
    /// Evaluate `expand(expr)` eagerly.
    EvaluateExpand { expand_call: ExprId },
    /// Unwrap internal hold wrapper.
    UnwrapHold { inner: ExprId },
    /// Keep expression as-is.
    Keep,
}

/// Classify whether an expression needs runtime pre-evaluation before GCD paths.
///
/// Rules:
/// - `expand(...)` (builtin) -> [`GcdPreEvalDirective::EvaluateExpand`]
/// - `__hold(x)` -> [`GcdPreEvalDirective::UnwrapHold`]
/// - `factor(...)` / `simplify(...)` -> [`GcdPreEvalDirective::Keep`]
pub fn classify_pre_evaluate_for_gcd(ctx: &Context, expr: ExprId) -> GcdPreEvalDirective {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return GcdPreEvalDirective::Keep;
    };

    if matches!(ctx.builtin_of(*fn_id), Some(cas_ast::BuiltinFn::Expand)) {
        return GcdPreEvalDirective::EvaluateExpand { expand_call: expr };
    }

    if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Hold) && !args.is_empty() {
        return GcdPreEvalDirective::UnwrapHold { inner: args[0] };
    }

    let name = ctx.sym_name(*fn_id);
    if name == "factor" || name == "simplify" {
        return GcdPreEvalDirective::Keep;
    }

    GcdPreEvalDirective::Keep
}

/// Apply runtime pre-evaluation policy for one GCD operand.
///
/// This centralizes the common handling of:
/// - `expand(...)` wrappers (delegated to caller-provided evaluator),
/// - `__hold(...)` unwrapping,
/// - passthrough for all other expressions.
pub fn pre_evaluate_for_gcd_with<FEvalExpand>(
    ctx: &mut Context,
    expr: ExprId,
    mut eval_expand: FEvalExpand,
) -> ExprId
where
    FEvalExpand: FnMut(&mut Context, ExprId) -> ExprId,
{
    match classify_pre_evaluate_for_gcd(ctx, expr) {
        GcdPreEvalDirective::EvaluateExpand { expand_call } => eval_expand(ctx, expand_call),
        GcdPreEvalDirective::UnwrapHold { inner } => inner,
        GcdPreEvalDirective::Keep => expr,
    }
}

/// The primitive-normalised gcd of two single-variable polynomials over ℚ via
/// `Polynomial::gcd` (Euclidean), or `None` if either side is not a polynomial in one
/// shared variable. Gives the clean `x-1`/`x+1` form rather than an arbitrary rational
/// scale, and `1` exactly when the two polynomials are coprime.
fn try_univariate_euclidean_gcd(ctx: &mut Context, a: ExprId, b: ExprId) -> Option<ExprId> {
    let mut vars = cas_ast::collect_variables(ctx, a);
    vars.extend(cas_ast::collect_variables(ctx, b));
    if vars.len() != 1 {
        return None;
    }
    let var = vars.into_iter().next()?;
    let pa = crate::polynomial::Polynomial::from_expr(ctx, a, &var).ok()?;
    let pb = crate::polynomial::Polynomial::from_expr(ctx, b, &var).ok()?;
    if pa.is_zero() || pb.is_zero() {
        return None;
    }
    // Make the gcd monic. `Polynomial::gcd` normalises by content, but `content()` returns
    // `1` for non-integer coefficients, so an arbitrary rational scale (`5/4·(x+1)`) can
    // survive; dividing by the leading coefficient gives the canonical monic gcd (`x+1`).
    let g = pa.gcd(&pb);
    let lc = g.leading_coeff();
    let g = if num_traits::Zero::is_zero(&lc) || num_traits::One::is_one(&lc) {
        g
    } else {
        g.div_scalar(&lc)
    };
    Some(g.to_expr(ctx))
}

/// Compute polynomial GCD using the selected mode.
///
/// `pre_evaluate` lets runtime crates unwrap/evaluate wrappers before exact/modp paths.
/// `render` is used only for human-readable descriptions.
#[allow(clippy::too_many_arguments)]
pub fn compute_poly_gcd_unified_with<FPreEval, FRender>(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    goal: GcdGoal,
    mode: GcdMode,
    modp_preset: Option<ZippelPreset>,
    modp_main_var: Option<usize>,
    mut pre_evaluate: FPreEval,
    mut render: FRender,
) -> (ExprId, String)
where
    FPreEval: FnMut(&mut Context, ExprId) -> ExprId,
    FRender: FnMut(&Context, ExprId) -> String,
{
    match mode {
        GcdMode::Structural => {
            let gcd = poly_gcd_structural(ctx, a, b);
            // Structural matching only finds shared `Mul` factors, so it returns `1` for an
            // EXPANDED pair that genuinely shares a factor (`gcd(x^3-1, x^2-1)` -> `1`
            // instead of `x-1`). When it finds nothing, fall back to a correct gcd: the
            // primitive-normalised univariate Euclidean `Polynomial::gcd` (clean `x-1`, not
            // a `5/4·x+5/4` scale), and the exact gcd for multivariate inputs. `1` survives
            // only when the polynomials are genuinely coprime.
            let gcd = if matches!(ctx.get(gcd), Expr::Number(n) if n.is_one()) {
                if let Some(univariate) = try_univariate_euclidean_gcd(ctx, a, b) {
                    univariate
                } else {
                    let eval_a = pre_evaluate(ctx, a);
                    let eval_b = pre_evaluate(ctx, b);
                    let exact = gcd_exact(ctx, eval_a, eval_b, &GcdExactBudget::default()).gcd;
                    // The exact layers return `1` both for coprime inputs and when
                    // their multivariate heuristics miss an existing common factor
                    // (`gcd(x²−y², x²+2xy+y²)` → `1` instead of `x+y`). For the user
                    // gcd — but NOT fraction cancellation, which stays conservative so
                    // it never drops a removable-pole domain condition — confirm with a
                    // verified modular GCD before trusting that `1` means coprime.
                    if matches!(ctx.get(exact), Expr::Number(n) if n.is_one())
                        && goal != GcdGoal::CancelFraction
                    {
                        try_verified_modp_multivar_gcd(ctx, eval_a, eval_b).unwrap_or(exact)
                    } else {
                        exact
                    }
                }
            } else {
                gcd
            };
            let desc = format!("poly_gcd({}, {})", render(ctx, a), render(ctx, b));
            (gcd, desc)
        }
        GcdMode::Exact => {
            let eval_a = pre_evaluate(ctx, a);
            let eval_b = pre_evaluate(ctx, b);
            let budget = GcdExactBudget::default();
            let result = gcd_exact(ctx, eval_a, eval_b, &budget);
            let desc = format!(
                "poly_gcd({}, {}, exact) [{}]",
                render(ctx, a),
                render(ctx, b),
                format!("{:?}", result.layer_used).to_lowercase()
            );
            (result.gcd, desc)
        }
        GcdMode::Modp => {
            if goal == GcdGoal::CancelFraction {
                let one = ctx.num(1);
                return (
                    one,
                    "poly_gcd(..., modp) [blocked for soundness]".to_string(),
                );
            }

            let eval_a = pre_evaluate(ctx, a);
            let eval_b = pre_evaluate(ctx, b);
            let preset = modp_preset.unwrap_or(ZippelPreset::Aggressive);
            match compute_gcd_modp_expr_with_options(
                ctx,
                eval_a,
                eval_b,
                DEFAULT_PRIME,
                modp_main_var,
                Some(preset),
            ) {
                Ok(result) => (
                    result,
                    format!(
                        "poly_gcd({}, {}, modp) [{:?}]",
                        render(ctx, a),
                        render(ctx, b),
                        preset
                    ),
                ),
                Err(_) => {
                    let one = ctx.num(1);
                    (one, "poly_gcd(..., modp) [error]".to_string())
                }
            }
        }
        GcdMode::Auto => {
            let structural_gcd = poly_gcd_structural(ctx, a, b);
            let is_one = matches!(ctx.get(structural_gcd), Expr::Number(n) if n.is_one());
            if !is_one {
                return (
                    structural_gcd,
                    format!(
                        "poly_gcd({}, {}, auto) [structural]",
                        render(ctx, a),
                        render(ctx, b)
                    ),
                );
            }

            let eval_a = pre_evaluate(ctx, a);
            let eval_b = pre_evaluate(ctx, b);
            let budget = GcdExactBudget::default();
            let exact_result = gcd_exact(ctx, eval_a, eval_b, &budget);
            if exact_result.layer_used != GcdExactLayer::BudgetExceeded {
                return (
                    exact_result.gcd,
                    format!(
                        "poly_gcd({}, {}, auto) [exact:{:?}]",
                        render(ctx, a),
                        render(ctx, b),
                        exact_result.layer_used
                    ),
                );
            }

            if goal == GcdGoal::CancelFraction {
                let one = ctx.num(1);
                return (
                    one,
                    "poly_gcd(..., auto) [exact exceeded budget, modp blocked for soundness]"
                        .to_string(),
                );
            }

            let preset = modp_preset.unwrap_or(ZippelPreset::Aggressive);
            match compute_gcd_modp_expr_with_options(
                ctx,
                eval_a,
                eval_b,
                DEFAULT_PRIME,
                modp_main_var,
                Some(preset),
            ) {
                Ok(result) => (
                    result,
                    format!(
                        "poly_gcd({}, {}, auto) [modp:{:?} - probabilistic]",
                        render(ctx, a),
                        render(ctx, b),
                        preset
                    ),
                ),
                Err(_) => {
                    let one = ctx.num(1);
                    (one, "poly_gcd(..., auto) [modp error]".to_string())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        classify_pre_evaluate_for_gcd, compute_poly_gcd_unified_with, pre_evaluate_for_gcd_with,
        GcdPreEvalDirective,
    };
    use crate::gcd_zippel_modp::ZippelPreset;
    use crate::poly_gcd_mode::{GcdGoal, GcdMode};
    use cas_ast::{BuiltinFn, Expr};
    use cas_parser::parse;
    use num_traits::One;

    #[test]
    fn structural_mode_finds_common_factor() {
        let mut ctx = cas_ast::Context::new();
        let a = parse("(x+1)*(y+2)", &mut ctx).expect("parse a");
        let b = parse("(x+1)*(z+3)", &mut ctx).expect("parse b");
        let (gcd, desc) = compute_poly_gcd_unified_with(
            &mut ctx,
            a,
            b,
            GcdGoal::UserPolyGcd,
            GcdMode::Structural,
            None,
            None,
            |_ctx, id| id,
            |_ctx, id| format!("{id:?}"),
        );
        assert!(desc.contains("poly_gcd("));
        assert!(!matches!(ctx.get(gcd), Expr::Number(_)));
    }

    #[test]
    fn structural_mode_falls_back_to_euclidean_for_expanded_polynomials() {
        // Expanded pairs share no structural `Mul` factor, so the structural matcher returns
        // `1`; the Euclidean fallback recovers the correct monic gcd, and leaves `1` only
        // for genuinely coprime polynomials.
        for (lhs, rhs, expect) in [
            ("x^3-1", "x^2-1", "x - 1"),
            ("x^2-1", "x-1", "x - 1"),
            ("x^2-4", "x^2-x-2", "x - 2"),
            ("x^3-x^2-2*x", "x^3+x^2+x+1", "x + 1"), // was a 5/4·x+5/4 scale before monic norm
            ("x", "x+1", "1"),                       // coprime
            ("x^2+1", "x-1", "1"),                   // coprime
        ] {
            let mut ctx = cas_ast::Context::new();
            let a = parse(lhs, &mut ctx).expect("parse a");
            let b = parse(rhs, &mut ctx).expect("parse b");
            let (gcd, _desc) = compute_poly_gcd_unified_with(
                &mut ctx,
                a,
                b,
                GcdGoal::UserPolyGcd,
                GcdMode::Structural,
                None,
                None,
                |_ctx, id| id,
                |_ctx, id| format!("{id:?}"),
            );
            let rendered = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: gcd
                }
            );
            assert_eq!(rendered, expect, "gcd({lhs}, {rhs})");
        }
    }

    #[test]
    fn structural_mode_recovers_multivariate_gcd_via_verified_modp() {
        // The exact Layer 1/2/2.5 pipeline misses these genuine bivariate common
        // factors and returned `1` (a soundness wrong-answer: false coprimality).
        // The verified modular GCD recovers them, and exact-division verification
        // keeps coprime inputs at `1`.
        for (lhs, rhs, expect) in [
            ("x^2-y^2", "x^2+2*x*y+y^2", "x + y"), // (x-y)(x+y) vs (x+y)^2  → x+y
            ("(x+y)^2", "(x+y)*(x-y)", "x + y"),
            ("x+y", "x-y", "1"),     // genuinely coprime
            ("x^2+1", "x*y+1", "1"), // genuinely coprime, bivariate
            ("x*y", "y", "y"),       // monomial factor (Layer 1 path) still works
        ] {
            let mut ctx = cas_ast::Context::new();
            let a = parse(lhs, &mut ctx).expect("parse a");
            let b = parse(rhs, &mut ctx).expect("parse b");
            let (gcd, _desc) = compute_poly_gcd_unified_with(
                &mut ctx,
                a,
                b,
                GcdGoal::UserPolyGcd,
                GcdMode::Structural,
                None,
                None,
                |_ctx, id| id,
                |_ctx, id| format!("{id:?}"),
            );
            let rendered = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: gcd
                }
            );
            assert_eq!(rendered, expect, "gcd({lhs}, {rhs})");
        }
    }

    #[test]
    fn cancel_fraction_goal_does_not_use_modp_multivar_fallback() {
        // Fraction cancellation must stay conservative: a verified modular GCD is
        // sound for the gcd VALUE, but cancelling a polynomial factor from a
        // fraction can drop a removable-pole domain condition, so this goal keeps
        // the exact-only behaviour (`1` here) rather than recovering `x+y`.
        let mut ctx = cas_ast::Context::new();
        let a = parse("x^2-y^2", &mut ctx).expect("parse a");
        let b = parse("x^2+2*x*y+y^2", &mut ctx).expect("parse b");
        let (gcd, _desc) = compute_poly_gcd_unified_with(
            &mut ctx,
            a,
            b,
            GcdGoal::CancelFraction,
            GcdMode::Structural,
            None,
            None,
            |_ctx, id| id,
            |_ctx, id| format!("{id:?}"),
        );
        assert!(
            matches!(ctx.get(gcd), Expr::Number(n) if n.is_one()),
            "CancelFraction must not use the modp multivariate fallback"
        );
    }

    #[test]
    fn modp_mode_is_blocked_for_cancel_fraction_goal() {
        let mut ctx = cas_ast::Context::new();
        let a = parse("x^2+1", &mut ctx).expect("parse a");
        let b = parse("x+1", &mut ctx).expect("parse b");
        let (gcd, desc) = compute_poly_gcd_unified_with(
            &mut ctx,
            a,
            b,
            GcdGoal::CancelFraction,
            GcdMode::Modp,
            Some(ZippelPreset::Aggressive),
            None,
            |_ctx, id| id,
            |_ctx, id| format!("{id:?}"),
        );
        assert!(matches!(ctx.get(gcd), Expr::Number(_)));
        assert!(desc.contains("blocked for soundness"));
    }

    #[test]
    fn classify_pre_eval_marks_expand_and_hold() {
        let mut ctx = cas_ast::Context::new();
        let inner = parse("x+1", &mut ctx).expect("parse");
        let expand_call = ctx.call_builtin(BuiltinFn::Expand, vec![inner]);
        let hold = ctx.call_builtin(BuiltinFn::Hold, vec![inner]);

        assert!(matches!(
            classify_pre_evaluate_for_gcd(&ctx, expand_call),
            GcdPreEvalDirective::EvaluateExpand { .. }
        ));
        assert_eq!(
            classify_pre_evaluate_for_gcd(&ctx, hold),
            GcdPreEvalDirective::UnwrapHold { inner }
        );
    }

    #[test]
    fn explicit_composition_can_wrap_output_in_hold() {
        let mut ctx = cas_ast::Context::new();
        let a = parse("(x+1)*(y+2)", &mut ctx).expect("parse a");
        let b = parse("(x+1)*(z+3)", &mut ctx).expect("parse b");

        let (gcd, _desc) = compute_poly_gcd_unified_with(
            &mut ctx,
            a,
            b,
            GcdGoal::UserPolyGcd,
            GcdMode::Structural,
            None,
            None,
            |core_ctx, id| pre_evaluate_for_gcd_with(core_ctx, id, |_ctx, expand_call| expand_call),
            |_ctx, id| format!("{id:?}"),
        );
        let held = cas_ast::hold::wrap_hold(&mut ctx, gcd);

        assert!(cas_ast::hold::is_hold(&ctx, held));
    }

    #[test]
    fn classify_pre_eval_keeps_factor_and_plain() {
        let mut ctx = cas_ast::Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let factor = ctx.call("factor", vec![x]);
        assert_eq!(
            classify_pre_evaluate_for_gcd(&ctx, factor),
            GcdPreEvalDirective::Keep
        );
        assert_eq!(
            classify_pre_evaluate_for_gcd(&ctx, x),
            GcdPreEvalDirective::Keep
        );
    }

    #[test]
    fn pre_evaluate_for_gcd_with_unwraps_hold_and_keeps_plain() {
        let mut ctx = cas_ast::Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let hold = ctx.call_builtin(BuiltinFn::Hold, vec![x]);

        let mut calls = 0usize;
        let kept = pre_evaluate_for_gcd_with(&mut ctx, x, |_ctx, id| {
            calls += 1;
            id
        });
        let unwrapped = pre_evaluate_for_gcd_with(&mut ctx, hold, |_ctx, id| {
            calls += 1;
            id
        });

        assert_eq!(kept, x);
        assert_eq!(unwrapped, x);
        assert_eq!(calls, 0, "expand evaluator should not run");
    }

    #[test]
    fn pre_evaluate_for_gcd_with_routes_expand_to_callback() {
        let mut ctx = cas_ast::Context::new();
        let x = parse("x+1", &mut ctx).expect("parse");
        let expand_call = ctx.call_builtin(BuiltinFn::Expand, vec![x]);

        let mut seen = None;
        let out = pre_evaluate_for_gcd_with(&mut ctx, expand_call, |_ctx, call| {
            seen = Some(call);
            x
        });

        assert_eq!(seen, Some(expand_call));
        assert_eq!(out, x);
    }

    #[test]
    fn explicit_pre_eval_invokes_expand_callback_for_exact_mode() {
        let mut ctx = cas_ast::Context::new();
        let a = parse("expand((x+1)^2)", &mut ctx).expect("parse a");
        let b = parse("x+1", &mut ctx).expect("parse b");

        let mut expanded_calls = 0usize;
        let (_gcd, _desc) = compute_poly_gcd_unified_with(
            &mut ctx,
            a,
            b,
            GcdGoal::UserPolyGcd,
            GcdMode::Exact,
            None,
            None,
            |core_ctx, id| {
                pre_evaluate_for_gcd_with(core_ctx, id, |inner_ctx, expand_call| {
                    expanded_calls += 1;
                    let Expr::Function(_, args) = inner_ctx.get(expand_call) else {
                        panic!("expected expand call");
                    };
                    args[0]
                })
            },
            |_ctx, id| format!("{id:?}"),
        );
        assert_eq!(expanded_calls, 1);
    }
}
