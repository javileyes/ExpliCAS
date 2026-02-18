//! Shared helpers for mod-p polynomial call entry points.
//!
//! These helpers keep rule-layer code in `cas_engine` thin by centralizing
//! common mod-p call behavior in `cas_math`.

use crate::gcd_zippel_modp::ZippelPreset;
use crate::poly_modp_conv::{
    check_poly_equal_modp_expr, compute_gcd_modp_expr_with_factor_extraction, PolyConvError,
};
use crate::poly_store::{compute_poly_mul_modp_meta, PolyMeta, PolyMulMetaError};
use cas_ast::{hold, Context, ExprId};

/// Compute `poly_gcd_modp` with structural factor extraction and wrap in `__hold(...)`.
pub fn compute_gcd_modp_held_expr(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    p: u64,
    main_var: Option<usize>,
    preset: Option<ZippelPreset>,
) -> Result<ExprId, PolyConvError> {
    let gcd = compute_gcd_modp_expr_with_factor_extraction(ctx, a, b, p, main_var, preset)?;
    Ok(hold::wrap_hold(ctx, gcd))
}

/// Compute `poly_eq_modp(a, b, p)` and return boolean result.
pub fn compute_poly_eq_modp(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    p: u64,
) -> Result<bool, PolyConvError> {
    check_poly_equal_modp_expr(ctx, a, b, p)
}

/// Compute `poly_eq_modp(a, b, p)` and return numeric indicator expression plus bool.
pub fn compute_poly_eq_modp_indicator(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    p: u64,
) -> Result<(ExprId, bool), PolyConvError> {
    let equal = compute_poly_eq_modp(ctx, a, b, p)?;
    let indicator = if equal { ctx.num(1) } else { ctx.num(0) };
    Ok((indicator, equal))
}

/// Compute metadata for `poly_mul_modp(a, b, p)`.
pub fn compute_poly_mul_modp_stats(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    p: u64,
    max_store_terms: usize,
) -> Result<PolyMeta, PolyMulMetaError> {
    compute_poly_mul_modp_meta(ctx, a, b, p, max_store_terms)
}

/// Build `poly_mul_stats(terms, degree, vars, modulus)` expression from metadata.
pub fn build_poly_mul_stats_expr(ctx: &mut Context, meta: &PolyMeta) -> ExprId {
    let terms = ctx.num(meta.n_terms as i64);
    let degree = ctx.num(meta.max_total_degree as i64);
    let nvars = ctx.num(meta.n_vars as i64);
    let modulus = ctx.num(meta.modulus as i64);
    ctx.call("poly_mul_stats", vec![terms, degree, nvars, modulus])
}
