//! poly_gcd_modp and poly_eq_modp REPL functions.
//!
//! Exposes Zippel mod-p GCD to REPL for fast polynomial verification.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::poly_modp_calls::{
    try_eval_poly_eq_modp_call_with_error_policy, try_eval_poly_gcd_modp_call_with_error_policy,
    PolyGcdModpEvalPath,
};
use cas_math::poly_modp_conv::DEFAULT_PRIME;

fn format_poly_gcd_modp_desc(
    ctx: &cas_ast::Context,
    a_expr: cas_ast::ExprId,
    b_expr: cas_ast::ExprId,
    path: PolyGcdModpEvalPath,
) -> String {
    let render = |id| cas_formatter::render_expr(ctx, id);
    match path {
        PolyGcdModpEvalPath::FastDefault => format!(
            "poly_gcd_modp({}, {}) [eager eval + factor extraction]",
            render(a_expr),
            render(b_expr)
        ),
        PolyGcdModpEvalPath::ExplicitOptions => {
            format!("poly_gcd_modp({}, {})", render(a_expr), render(b_expr))
        }
    }
}

fn format_poly_eq_modp_desc(
    ctx: &cas_ast::Context,
    a_expr: cas_ast::ExprId,
    b_expr: cas_ast::ExprId,
    equal: bool,
) -> String {
    format!(
        "poly_eq_modp({}, {}) = {}",
        cas_formatter::render_expr(ctx, a_expr),
        cas_formatter::render_expr(ctx, b_expr),
        if equal { "true" } else { "false" }
    )
}

// Rule for poly_gcd_modp(a, b [, p]) function.
// Computes Zippel GCD of two polynomial expressions mod p.
define_rule!(
    PolyGcdModpRule,
    "Polynomial GCD mod p",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    priority: 200, // High priority to evaluate early
    |ctx, expr| {
        let call = try_eval_poly_gcd_modp_call_with_error_policy(ctx, expr, DEFAULT_PRIME, |_err| {})?;
        let desc = format_poly_gcd_modp_desc(ctx, call.a_expr, call.b_expr, call.path);
        Some(Rewrite::simple(call.held_expr, desc))
    }
);

// Rule for poly_eq_modp(a, b [, p]) function.
// Returns 1 if polynomials are equal mod p, 0 otherwise.
define_rule!(
    PolyEqModpRule,
    "Polynomial equality mod p",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    priority: 200,
    |ctx, expr| {
        let call = try_eval_poly_eq_modp_call_with_error_policy(ctx, expr, DEFAULT_PRIME, |_err| {})?;
        let desc = format_poly_eq_modp_desc(ctx, call.a_expr, call.b_expr, call.equal);
        Some(Rewrite::simple(call.indicator_expr, desc))
    }
);
