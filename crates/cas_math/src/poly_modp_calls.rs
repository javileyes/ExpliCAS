//! Shared helpers for mod-p polynomial call entry points.
//!
//! These helpers keep rule-layer code in `cas_engine` thin by centralizing
//! common mod-p call behavior in `cas_math`.

use crate::expr_extract::extract_u64_integer;
use crate::gcd_zippel_modp::ZippelPreset;
use crate::poly_gcd_mode::parse_modp_options;
use crate::poly_modp_conv::{
    check_poly_equal_modp_expr, compute_gcd_modp_expr_with_factor_extraction, strip_hold,
    PolyConvError, DEFAULT_PRIME,
};
use crate::poly_store::{compute_poly_mul_modp_meta, PolyMeta, PolyMulMetaError};
use cas_ast::{hold, Context, Expr, ExprId};

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

/// Rewrite helper for held polynomial subtraction:
/// `__hold(P) - __hold(Q) -> 0` when `P == Q` in mod-p polynomial space.
///
/// Returns `Some(0)` only when:
/// - expression is `Sub(a, b)`
/// - at least one side is `__hold(...)`
/// - both normalized polynomial forms are equal mod `p`
pub fn try_rewrite_hold_poly_sub_to_zero(
    ctx: &mut Context,
    expr: ExprId,
    p: u64,
) -> Option<ExprId> {
    let Expr::Sub(a, b) = ctx.get(expr) else {
        return None;
    };
    let (a, b) = (*a, *b);

    let a_is_hold = hold::is_hold(ctx, a);
    let b_is_hold = hold::is_hold(ctx, b);
    if !(a_is_hold || b_is_hold) {
        return None;
    }

    let a_inner = strip_hold(ctx, a);
    let b_inner = strip_hold(ctx, b);
    if check_poly_equal_modp_expr(ctx, a_inner, b_inner, p).ok()? {
        return Some(ctx.num(0));
    }
    None
}

/// Parsed and evaluated payload for one `poly_mul_modp(...)` function call.
#[derive(Debug, Clone)]
pub struct PolyMulModpStatsCall {
    pub a_expr: ExprId,
    pub b_expr: ExprId,
    pub modulus: u64,
    pub meta: PolyMeta,
    pub stats_expr: ExprId,
}

/// Evaluation path used for a successful `poly_gcd_modp` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyGcdModpEvalPath {
    FastDefault,
    ExplicitOptions,
}

/// Parsed and evaluated payload for one `poly_gcd_modp(...)` / `pgcdp(...)` call.
#[derive(Debug, Clone)]
pub struct PolyGcdModpCall {
    pub a_expr: ExprId,
    pub b_expr: ExprId,
    pub held_expr: ExprId,
    pub path: PolyGcdModpEvalPath,
}

/// Parsed and evaluated payload for one `poly_eq_modp(...)` / `peqp(...)` call.
#[derive(Debug, Clone)]
pub struct PolyEqModpCall {
    pub a_expr: ExprId,
    pub b_expr: ExprId,
    pub modulus: u64,
    pub indicator_expr: ExprId,
    pub equal: bool,
}

/// Try to parse and evaluate `poly_mul_modp(a, b [, p])`.
///
/// Returns:
/// - `Ok(None)` when expression is not a valid `poly_mul_modp` call
/// - `Ok(Some(...))` with computed stats payload when successful
/// - `Err(...)` when conversion/size checks fail during poly analysis
pub fn try_eval_poly_mul_modp_stats_call(
    ctx: &mut Context,
    expr: ExprId,
    default_prime: u64,
    max_store_terms: usize,
) -> Result<Option<PolyMulModpStatsCall>, PolyMulMetaError> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return Ok(None);
    };
    let fn_id = *fn_id;
    let args = args.clone();

    if ctx.sym_name(fn_id) != "poly_mul_modp" {
        return Ok(None);
    }
    if args.len() < 2 || args.len() > 3 {
        return Ok(None);
    }

    let a_expr = args[0];
    let b_expr = args[1];
    let modulus = if args.len() == 3 {
        let Some(p) = extract_u64_integer(ctx, args[2]) else {
            return Ok(None);
        };
        p
    } else {
        default_prime
    };

    let meta = compute_poly_mul_modp_stats(ctx, a_expr, b_expr, modulus, max_store_terms)?;
    let stats_expr = build_poly_mul_stats_expr(ctx, &meta);
    Ok(Some(PolyMulModpStatsCall {
        a_expr,
        b_expr,
        modulus,
        meta,
        stats_expr,
    }))
}

/// Evaluate `poly_mul_modp(...)` and absorb common error policy:
/// - conversion failures are ignored (`None`),
/// - size over-limit triggers callback and returns `None`.
pub fn try_eval_poly_mul_modp_stats_call_with_limit_policy<FOnEstimatedTooLarge>(
    ctx: &mut Context,
    expr: ExprId,
    default_prime: u64,
    max_store_terms: usize,
    mut on_estimated_too_large: FOnEstimatedTooLarge,
) -> Option<PolyMulModpStatsCall>
where
    FOnEstimatedTooLarge: FnMut(u128, usize),
{
    match try_eval_poly_mul_modp_stats_call(ctx, expr, default_prime, max_store_terms) {
        Ok(Some(call)) => Some(call),
        Ok(None) => None,
        Err(PolyMulMetaError::ConversionFailed) => None,
        Err(PolyMulMetaError::EstimatedTooLarge {
            estimated_terms,
            limit,
        }) => {
            on_estimated_too_large(estimated_terms, limit);
            None
        }
    }
}

/// Try to parse and evaluate `poly_gcd_modp(a, b [, options...])`.
///
/// Returns:
/// - `Ok(None)` when expression is not a valid `poly_gcd_modp`/`pgcdp` call
/// - `Ok(Some(...))` with computed held gcd payload when successful
/// - `Err(...)` when polynomial conversion/evaluation fails
pub fn try_eval_poly_gcd_modp_call(
    ctx: &mut Context,
    expr: ExprId,
    default_prime: u64,
) -> Result<Option<PolyGcdModpCall>, PolyConvError> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return Ok(None);
    };
    let fn_id = *fn_id;
    let args = args.clone();

    let name = ctx.sym_name(fn_id);
    if name != "poly_gcd_modp" && name != "pgcdp" {
        return Ok(None);
    }
    if args.len() < 2 || args.len() > 4 {
        return Ok(None);
    }

    let a_expr = args[0];
    let b_expr = args[1];

    if let Ok(held_expr) =
        compute_gcd_modp_held_expr(ctx, a_expr, b_expr, default_prime, None, None)
    {
        return Ok(Some(PolyGcdModpCall {
            a_expr,
            b_expr,
            held_expr,
            path: PolyGcdModpEvalPath::FastDefault,
        }));
    }

    let (preset, main_var) = parse_modp_options(ctx, &args[2..]);
    let held_expr =
        compute_gcd_modp_held_expr(ctx, a_expr, b_expr, default_prime, main_var, preset)?;
    Ok(Some(PolyGcdModpCall {
        a_expr,
        b_expr,
        held_expr,
        path: PolyGcdModpEvalPath::ExplicitOptions,
    }))
}

/// Evaluate `poly_gcd_modp` and apply a caller-defined error policy.
///
/// Returns `None` for non-matching calls or when an evaluation error occurs.
pub fn try_eval_poly_gcd_modp_call_with_error_policy<FOnError>(
    ctx: &mut Context,
    expr: ExprId,
    default_prime: u64,
    mut on_error: FOnError,
) -> Option<PolyGcdModpCall>
where
    FOnError: FnMut(&PolyConvError),
{
    match try_eval_poly_gcd_modp_call(ctx, expr, default_prime) {
        Ok(Some(call)) => Some(call),
        Ok(None) => None,
        Err(err) => {
            on_error(&err);
            None
        }
    }
}

/// Rewrite `poly_gcd_modp(...)` call into `(held_result_expr, description)`.
///
/// Returns `None` when:
/// - expression does not match `poly_gcd_modp`/`pgcdp`, or
/// - evaluation fails (error forwarded to `on_error`).
pub fn rewrite_poly_gcd_modp_call_with<FOnError, FRender>(
    ctx: &mut Context,
    expr: ExprId,
    default_prime: u64,
    on_error: FOnError,
    mut render_expr: FRender,
) -> Option<(ExprId, String)>
where
    FOnError: FnMut(&PolyConvError),
    FRender: FnMut(&Context, ExprId) -> String,
{
    let call = try_eval_poly_gcd_modp_call_with_error_policy(ctx, expr, default_prime, on_error)?;
    let desc = format_poly_gcd_modp_desc_with(call.a_expr, call.b_expr, call.path, |id| {
        render_expr(ctx, id)
    });
    Some((call.held_expr, desc))
}

/// Try to parse and evaluate `poly_eq_modp(a, b [, p])`.
///
/// Returns:
/// - `Ok(None)` when expression is not a valid `poly_eq_modp`/`peqp` call
/// - `Ok(Some(...))` with indicator payload when successful
/// - `Err(...)` when polynomial conversion/evaluation fails
pub fn try_eval_poly_eq_modp_call(
    ctx: &mut Context,
    expr: ExprId,
    default_prime: u64,
) -> Result<Option<PolyEqModpCall>, PolyConvError> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return Ok(None);
    };
    let fn_id = *fn_id;
    let args = args.clone();

    let name = ctx.sym_name(fn_id);
    if name != "poly_eq_modp" && name != "peqp" {
        return Ok(None);
    }
    if args.len() < 2 || args.len() > 3 {
        return Ok(None);
    }

    let a_expr = args[0];
    let b_expr = args[1];
    let modulus = if args.len() == 3 {
        extract_u64_integer(ctx, args[2]).unwrap_or(default_prime)
    } else {
        default_prime
    };

    let (indicator_expr, equal) = compute_poly_eq_modp_indicator(ctx, a_expr, b_expr, modulus)?;
    Ok(Some(PolyEqModpCall {
        a_expr,
        b_expr,
        modulus,
        indicator_expr,
        equal,
    }))
}

/// Evaluate `poly_eq_modp` and apply a caller-defined error policy.
///
/// Returns `None` for non-matching calls or when an evaluation error occurs.
pub fn try_eval_poly_eq_modp_call_with_error_policy<FOnError>(
    ctx: &mut Context,
    expr: ExprId,
    default_prime: u64,
    mut on_error: FOnError,
) -> Option<PolyEqModpCall>
where
    FOnError: FnMut(&PolyConvError),
{
    match try_eval_poly_eq_modp_call(ctx, expr, default_prime) {
        Ok(Some(call)) => Some(call),
        Ok(None) => None,
        Err(err) => {
            on_error(&err);
            None
        }
    }
}

/// Rewrite `poly_eq_modp(...)` call into `(indicator_expr, description)`.
///
/// Returns `None` when:
/// - expression does not match `poly_eq_modp`/`peqp`, or
/// - evaluation fails (error forwarded to `on_error`).
pub fn rewrite_poly_eq_modp_call_with<FOnError, FRender>(
    ctx: &mut Context,
    expr: ExprId,
    default_prime: u64,
    on_error: FOnError,
    mut render_expr: FRender,
) -> Option<(ExprId, String)>
where
    FOnError: FnMut(&PolyConvError),
    FRender: FnMut(&Context, ExprId) -> String,
{
    let call = try_eval_poly_eq_modp_call_with_error_policy(ctx, expr, default_prime, on_error)?;
    let desc = format_poly_eq_modp_desc_with(call.a_expr, call.b_expr, call.equal, |id| {
        render_expr(ctx, id)
    });
    Some((call.indicator_expr, desc))
}

/// Rewrite `poly_mul_modp(...)` call into `(stats_expr, description)`.
///
/// Returns `None` when expression is non-matching or conversion fails.
/// If estimated size exceeds `max_store_terms`, callback is invoked and `None` is returned.
pub fn rewrite_poly_mul_modp_stats_call_with_limit_policy<FOnEstimatedTooLarge>(
    ctx: &mut Context,
    expr: ExprId,
    default_prime: u64,
    max_store_terms: usize,
    on_estimated_too_large: FOnEstimatedTooLarge,
) -> Option<(ExprId, String)>
where
    FOnEstimatedTooLarge: FnMut(u128, usize),
{
    let call = try_eval_poly_mul_modp_stats_call_with_limit_policy(
        ctx,
        expr,
        default_prime,
        max_store_terms,
        on_estimated_too_large,
    )?;
    let desc = format_poly_mul_modp_stats_desc(&call.meta, call.modulus);
    Some((call.stats_expr, desc))
}

/// Build `poly_mul_stats(terms, degree, vars, modulus)` expression from metadata.
pub fn build_poly_mul_stats_expr(ctx: &mut Context, meta: &PolyMeta) -> ExprId {
    let terms = ctx.num(meta.n_terms as i64);
    let degree = ctx.num(meta.max_total_degree as i64);
    let nvars = ctx.num(meta.n_vars as i64);
    let modulus = ctx.num(meta.modulus as i64);
    ctx.call("poly_mul_stats", vec![terms, degree, nvars, modulus])
}

/// Build human-readable `poly_mul_modp` stats description.
pub fn format_poly_mul_modp_stats_desc(meta: &PolyMeta, modulus: u64) -> String {
    format!(
        "poly_mul_modp: {} terms, degree {}, {} vars (mod {})",
        meta.n_terms, meta.max_total_degree, meta.n_vars, modulus
    )
}

/// Build human-readable `poly_gcd_modp` description.
pub fn format_poly_gcd_modp_desc_with<FRender>(
    a_expr: ExprId,
    b_expr: ExprId,
    path: PolyGcdModpEvalPath,
    mut render: FRender,
) -> String
where
    FRender: FnMut(ExprId) -> String,
{
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

/// Build human-readable `poly_eq_modp` description.
pub fn format_poly_eq_modp_desc_with<FRender>(
    a_expr: ExprId,
    b_expr: ExprId,
    equal: bool,
    mut render: FRender,
) -> String
where
    FRender: FnMut(ExprId) -> String,
{
    format!(
        "poly_eq_modp({}, {}) = {}",
        render(a_expr),
        render(b_expr),
        if equal { "true" } else { "false" }
    )
}

/// Eagerly evaluate `poly_gcd_modp` calls in an expression tree.
///
/// Traversal is top-down: when a `poly_gcd_modp`/`pgcdp` call is evaluated,
/// traversal stops for that subtree (children are not visited).
///
/// `include_items` controls whether callback items are collected in the returned vector.
pub fn eager_eval_poly_gcd_calls_with<Item, FBuildItem>(
    ctx: &mut Context,
    expr: ExprId,
    include_items: bool,
    mut build_item: FBuildItem,
) -> (ExprId, Vec<Item>)
where
    FBuildItem: FnMut(&Context, ExprId, ExprId) -> Item,
{
    let mut items = Vec::new();
    let result =
        eager_eval_poly_gcd_recursive(ctx, expr, include_items, &mut items, &mut build_item);
    (result, items)
}

fn eager_eval_poly_gcd_recursive<Item, FBuildItem>(
    ctx: &mut Context,
    expr: ExprId,
    include_items: bool,
    items: &mut Vec<Item>,
    build_item: &mut FBuildItem,
) -> ExprId
where
    FBuildItem: FnMut(&Context, ExprId, ExprId) -> Item,
{
    if let Some(rewritten) = try_eval_poly_gcd_call(ctx, expr) {
        if include_items {
            items.push(build_item(ctx, expr, rewritten));
        }
        return rewritten;
    }

    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let fn_id = *fn_id;
        let args = args.clone();
        let new_args: Vec<ExprId> = args
            .iter()
            .map(|&arg| eager_eval_poly_gcd_recursive(ctx, arg, include_items, items, build_item))
            .collect();
        if new_args
            .iter()
            .zip(args.iter())
            .any(|(new, old)| new != old)
        {
            return ctx.add(Expr::Function(fn_id, new_args));
        }
        return expr;
    }

    enum Recurse {
        Binary(ExprId, ExprId, u8), // 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Pow
        Unary(ExprId, u8),          // 0=Neg, 1=Hold
        Leaf,
    }

    let recurse = match ctx.get(expr) {
        Expr::Add(l, r) => Recurse::Binary(*l, *r, 0),
        Expr::Sub(l, r) => Recurse::Binary(*l, *r, 1),
        Expr::Mul(l, r) => Recurse::Binary(*l, *r, 2),
        Expr::Div(l, r) => Recurse::Binary(*l, *r, 3),
        Expr::Pow(b, e) => Recurse::Binary(*b, *e, 4),
        Expr::Neg(inner) => Recurse::Unary(*inner, 0),
        Expr::Hold(inner) => Recurse::Unary(*inner, 1),
        _ => Recurse::Leaf,
    };

    match recurse {
        Recurse::Binary(l, r, op) => {
            let nl = eager_eval_poly_gcd_recursive(ctx, l, include_items, items, build_item);
            let nr = eager_eval_poly_gcd_recursive(ctx, r, include_items, items, build_item);
            if nl != l || nr != r {
                match op {
                    0 => ctx.add(Expr::Add(nl, nr)),
                    1 => ctx.add(Expr::Sub(nl, nr)),
                    2 => ctx.add(Expr::Mul(nl, nr)),
                    3 => ctx.add(Expr::Div(nl, nr)),
                    _ => ctx.add(Expr::Pow(nl, nr)),
                }
            } else {
                expr
            }
        }
        Recurse::Unary(inner, op) => {
            let ni = eager_eval_poly_gcd_recursive(ctx, inner, include_items, items, build_item);
            if ni != inner {
                match op {
                    0 => ctx.add(Expr::Neg(ni)),
                    _ => ctx.add(Expr::Hold(ni)),
                }
            } else {
                expr
            }
        }
        Recurse::Leaf => expr,
    }
}

fn try_eval_poly_gcd_call(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let fn_id = *fn_id;
        let args = args.clone();
        let name = ctx.sym_name(fn_id);
        if (name == "poly_gcd_modp" || name == "pgcdp") && args.len() >= 2 {
            return compute_gcd_modp_held_expr(ctx, args[0], args[1], DEFAULT_PRIME, None, None)
                .ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;
    use cas_parser::parse;

    #[test]
    fn eq_indicator_returns_true_and_one_for_equivalent_polys() {
        let mut ctx = Context::new();
        let a = parse("x + 1", &mut ctx).expect("parse a");
        let b = parse("1 + x", &mut ctx).expect("parse b");

        let (indicator, equal) = compute_poly_eq_modp_indicator(&mut ctx, a, b, 101).expect("eq");
        assert!(equal);
        assert_eq!(
            ctx.get(indicator),
            &Expr::Number(num_rational::BigRational::from_integer(1.into()))
        );
    }

    #[test]
    fn gcd_result_is_wrapped_in_hold() {
        let mut ctx = Context::new();
        let a = parse("x^2 + 1", &mut ctx).expect("parse a");
        let b = parse("x + 1", &mut ctx).expect("parse b");

        let held = compute_gcd_modp_held_expr(&mut ctx, a, b, 101, None, None).expect("gcd");
        assert!(cas_ast::hold::is_hold(&ctx, held));
    }

    #[test]
    fn poly_mul_stats_expr_shape() {
        let mut ctx = Context::new();
        let meta = PolyMeta {
            modulus: 101,
            n_terms: 12,
            n_vars: 3,
            max_total_degree: 7,
            var_names: vec!["x".to_string(), "y".to_string(), "z".to_string()],
        };
        let stats = build_poly_mul_stats_expr(&mut ctx, &meta);

        if let Expr::Function(fn_id, args) = ctx.get(stats) {
            assert_eq!(ctx.sym_name(*fn_id), "poly_mul_stats");
            assert_eq!(args.len(), 4);
        } else {
            panic!("expected poly_mul_stats function");
        }
    }

    #[test]
    fn format_poly_mul_modp_stats_desc_includes_core_fields() {
        let meta = PolyMeta {
            modulus: 101,
            n_terms: 12,
            n_vars: 3,
            max_total_degree: 7,
            var_names: vec!["x".to_string(), "y".to_string(), "z".to_string()],
        };
        let desc = format_poly_mul_modp_stats_desc(&meta, 101);
        assert!(desc.contains("12 terms"));
        assert!(desc.contains("degree 7"));
        assert!(desc.contains("3 vars"));
        assert!(desc.contains("mod 101"));
    }

    #[test]
    fn format_poly_gcd_modp_desc_with_renders_by_path() {
        let a = cas_ast::ExprId::from_raw(10);
        let b = cas_ast::ExprId::from_raw(11);
        let render = |id: cas_ast::ExprId| {
            if id.index() == a.index() {
                "A".to_string()
            } else {
                "B".to_string()
            }
        };
        let fast = format_poly_gcd_modp_desc_with(a, b, PolyGcdModpEvalPath::FastDefault, render);
        let explicit =
            format_poly_gcd_modp_desc_with(a, b, PolyGcdModpEvalPath::ExplicitOptions, render);
        assert!(fast.contains("eager eval + factor extraction"));
        assert!(fast.contains("A"));
        assert!(fast.contains("B"));
        assert!(explicit.contains("poly_gcd_modp(A, B)"));
    }

    #[test]
    fn format_poly_eq_modp_desc_with_renders_truth_value() {
        let a = cas_ast::ExprId::from_raw(20);
        let b = cas_ast::ExprId::from_raw(21);
        let render = |id: cas_ast::ExprId| {
            if id.index() == a.index() {
                "LHS".to_string()
            } else {
                "RHS".to_string()
            }
        };
        let yes = format_poly_eq_modp_desc_with(a, b, true, render);
        let no = format_poly_eq_modp_desc_with(a, b, false, render);
        assert!(yes.contains("poly_eq_modp(LHS, RHS)"));
        assert!(yes.ends_with("true"));
        assert!(no.ends_with("false"));
    }

    #[test]
    fn try_rewrite_hold_poly_sub_to_zero_rewrites_equal_held_polys() {
        let mut ctx = Context::new();
        let poly = parse("x + 1", &mut ctx).expect("parse poly");
        let hold_a = hold::wrap_hold(&mut ctx, poly);
        let hold_b = hold::wrap_hold(&mut ctx, poly);
        let sub_expr = ctx.add(Expr::Sub(hold_a, hold_b));

        let rewritten = try_rewrite_hold_poly_sub_to_zero(&mut ctx, sub_expr, DEFAULT_PRIME)
            .expect("expected rewrite");
        assert!(matches!(ctx.get(rewritten), Expr::Number(_)));
    }

    #[test]
    fn try_rewrite_hold_poly_sub_to_zero_ignores_non_hold_subtractions() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let sub_expr = ctx.add(Expr::Sub(x, y));
        assert!(try_rewrite_hold_poly_sub_to_zero(&mut ctx, sub_expr, DEFAULT_PRIME).is_none());
    }

    #[test]
    fn try_rewrite_hold_poly_sub_to_zero_explicit_default_prime_is_stable() {
        let mut ctx = Context::new();
        let poly = parse("x + 1", &mut ctx).expect("parse poly");
        let hold_a = hold::wrap_hold(&mut ctx, poly);
        let hold_b = hold::wrap_hold(&mut ctx, poly);
        let sub_expr = ctx.add(Expr::Sub(hold_a, hold_b));

        let rewritten = try_rewrite_hold_poly_sub_to_zero(&mut ctx, sub_expr, DEFAULT_PRIME);
        assert_eq!(rewritten, Some(ctx.num(0)));
    }

    #[test]
    fn try_eval_poly_mul_modp_stats_call_parses_and_computes() {
        let mut ctx = Context::new();
        let expr = parse("poly_mul_modp((x+1)^2, (x-1)^2)", &mut ctx).expect("parse call");
        let call = try_eval_poly_mul_modp_stats_call(&mut ctx, expr, DEFAULT_PRIME, 100_000)
            .expect("eval should not error")
            .expect("call should match");

        assert_eq!(call.modulus, DEFAULT_PRIME);
        if let Expr::Function(fn_id, args) = ctx.get(call.stats_expr) {
            assert_eq!(ctx.sym_name(*fn_id), "poly_mul_stats");
            assert_eq!(args.len(), 4);
        } else {
            panic!("expected poly_mul_stats expression");
        }
    }

    #[test]
    fn try_eval_poly_mul_modp_stats_call_rejects_non_numeric_modulus() {
        let mut ctx = Context::new();
        let expr = parse("poly_mul_modp(x+1, x+1, p)", &mut ctx).expect("parse call");
        let call = try_eval_poly_mul_modp_stats_call(&mut ctx, expr, DEFAULT_PRIME, 100_000)
            .expect("eval should not error");
        assert!(call.is_none());
    }

    #[test]
    fn try_eval_poly_mul_modp_stats_call_with_limit_policy_ignores_non_matching() {
        let mut ctx = Context::new();
        let expr = parse("x+1", &mut ctx).expect("parse expr");
        let mut warned = false;
        let call = try_eval_poly_mul_modp_stats_call_with_limit_policy(
            &mut ctx,
            expr,
            DEFAULT_PRIME,
            100_000,
            |_estimated, _limit| warned = true,
        );
        assert!(call.is_none());
        assert!(!warned);
    }

    #[test]
    fn try_eval_poly_mul_modp_stats_call_with_limit_policy_warns_on_estimate_over_limit() {
        let mut ctx = Context::new();
        let expr = parse("poly_mul_modp((x+1)^6, (x+1)^6)", &mut ctx).expect("parse call");
        let mut warned = false;
        let call = try_eval_poly_mul_modp_stats_call_with_limit_policy(
            &mut ctx,
            expr,
            DEFAULT_PRIME,
            1, // force overflow
            |_estimated, _limit| warned = true,
        );
        assert!(call.is_none());
        assert!(warned);
    }

    #[test]
    fn try_eval_poly_gcd_modp_call_matches_alias_and_returns_hold() {
        let mut ctx = Context::new();
        let expr = parse("pgcdp(x^2 + 2*x + 1, x + 1)", &mut ctx).expect("parse call");
        let call = try_eval_poly_gcd_modp_call(&mut ctx, expr, DEFAULT_PRIME)
            .expect("eval should not error")
            .expect("call should match");
        assert!(hold::is_hold(&ctx, call.held_expr));
    }

    #[test]
    fn try_eval_poly_gcd_modp_call_ignores_non_matching_function() {
        let mut ctx = Context::new();
        let expr = parse("foo(x)", &mut ctx).expect("parse call");
        let call = try_eval_poly_gcd_modp_call(&mut ctx, expr, DEFAULT_PRIME)
            .expect("eval should not error");
        assert!(call.is_none());
    }

    #[test]
    fn rewrite_poly_gcd_modp_call_with_returns_desc_and_hold() {
        let mut ctx = Context::new();
        let expr = parse("poly_gcd_modp(x^2+2*x+1, x+1)", &mut ctx).expect("parse");
        let rewritten = rewrite_poly_gcd_modp_call_with(
            &mut ctx,
            expr,
            DEFAULT_PRIME,
            |_e| {},
            |core_ctx, id| format!("{:?}", core_ctx.get(id)),
        )
        .expect("rewrite");
        assert!(hold::is_hold(&ctx, rewritten.0));
        assert!(rewritten.1.contains("poly_gcd_modp("));
    }

    #[test]
    fn rewrite_poly_eq_modp_call_with_returns_indicator_and_desc() {
        let mut ctx = Context::new();
        let expr = parse("poly_eq_modp(x+1, 1+x)", &mut ctx).expect("parse");
        let rewritten = rewrite_poly_eq_modp_call_with(
            &mut ctx,
            expr,
            DEFAULT_PRIME,
            |_e| {},
            |core_ctx, id| format!("{:?}", core_ctx.get(id)),
        )
        .expect("rewrite");
        assert!(matches!(ctx.get(rewritten.0), Expr::Number(_)));
        assert!(rewritten.1.contains("poly_eq_modp("));
    }

    #[test]
    fn rewrite_poly_gcd_modp_call_with_can_silence_errors_explicitly() {
        let mut ctx = Context::new();
        let expr = parse("poly_gcd_modp(x^2+2*x+1, x+1)", &mut ctx).expect("parse");
        let rewritten = rewrite_poly_gcd_modp_call_with(
            &mut ctx,
            expr,
            DEFAULT_PRIME,
            |_err| {},
            |core_ctx, id| format!("{:?}", core_ctx.get(id)),
        )
        .expect("rewrite");
        assert!(hold::is_hold(&ctx, rewritten.0));
        assert!(rewritten.1.contains("poly_gcd_modp("));
    }

    #[test]
    fn rewrite_poly_eq_modp_call_with_can_silence_errors_explicitly() {
        let mut ctx = Context::new();
        let expr = parse("poly_eq_modp(x+1, 1+x)", &mut ctx).expect("parse");
        let rewritten = rewrite_poly_eq_modp_call_with(
            &mut ctx,
            expr,
            DEFAULT_PRIME,
            |_err| {},
            |core_ctx, id| format!("{:?}", core_ctx.get(id)),
        )
        .expect("rewrite");
        assert!(matches!(ctx.get(rewritten.0), Expr::Number(_)));
        assert!(rewritten.1.contains("poly_eq_modp("));
    }

    #[test]
    fn rewrite_poly_mul_modp_stats_call_with_limit_policy_returns_stats_and_desc() {
        let mut ctx = Context::new();
        let expr = parse("poly_mul_modp((x+1)^2, (x-1)^2)", &mut ctx).expect("parse");
        let rewritten = rewrite_poly_mul_modp_stats_call_with_limit_policy(
            &mut ctx,
            expr,
            DEFAULT_PRIME,
            100_000,
            |_estimated, _limit| {},
        )
        .expect("rewrite");
        if let Expr::Function(fn_id, _args) = ctx.get(rewritten.0) {
            assert_eq!(ctx.sym_name(*fn_id), "poly_mul_stats");
        } else {
            panic!("expected poly_mul_stats");
        }
        assert!(rewritten.1.contains("poly_mul_modp:"));
    }

    #[test]
    fn rewrite_poly_mul_modp_stats_call_with_explicit_defaults_returns_stats_and_desc() {
        let mut ctx = Context::new();
        let expr = parse("poly_mul_modp((x+1)^2, (x-1)^2)", &mut ctx).expect("parse");
        let rewritten = rewrite_poly_mul_modp_stats_call_with_limit_policy(
            &mut ctx,
            expr,
            DEFAULT_PRIME,
            crate::poly_store::POLY_MAX_STORE_TERMS,
            |_estimated, _limit| {},
        )
        .expect("rewrite");
        if let Expr::Function(fn_id, _args) = ctx.get(rewritten.0) {
            assert_eq!(ctx.sym_name(*fn_id), "poly_mul_stats");
        } else {
            panic!("expected poly_mul_stats");
        }
        assert!(rewritten.1.contains("poly_mul_modp:"));
    }

    #[test]
    fn try_eval_poly_gcd_modp_call_with_error_policy_handles_non_match_without_logging() {
        let mut ctx = Context::new();
        let expr = parse("foo(x)", &mut ctx).expect("parse call");
        let mut logged = false;
        let call =
            try_eval_poly_gcd_modp_call_with_error_policy(&mut ctx, expr, DEFAULT_PRIME, |_err| {
                logged = true;
            });
        assert!(call.is_none());
        assert!(!logged);
    }

    #[test]
    fn try_eval_poly_eq_modp_call_returns_indicator_payload() {
        let mut ctx = Context::new();
        let expr = parse("poly_eq_modp(x+1, x+1)", &mut ctx).expect("parse call");
        let call = try_eval_poly_eq_modp_call(&mut ctx, expr, DEFAULT_PRIME)
            .expect("eval should not error")
            .expect("call should match");
        assert_eq!(call.modulus, DEFAULT_PRIME);
        assert!(call.equal);
        assert!(matches!(ctx.get(call.indicator_expr), Expr::Number(_)));
    }

    #[test]
    fn try_eval_poly_eq_modp_call_ignores_non_matching_function() {
        let mut ctx = Context::new();
        let expr = parse("bar(x)", &mut ctx).expect("parse call");
        let call = try_eval_poly_eq_modp_call(&mut ctx, expr, DEFAULT_PRIME)
            .expect("eval should not error");
        assert!(call.is_none());
    }

    #[test]
    fn try_eval_poly_eq_modp_call_with_error_policy_handles_non_match_without_logging() {
        let mut ctx = Context::new();
        let expr = parse("bar(x)", &mut ctx).expect("parse call");
        let mut logged = false;
        let call =
            try_eval_poly_eq_modp_call_with_error_policy(&mut ctx, expr, DEFAULT_PRIME, |_err| {
                logged = true;
            });
        assert!(call.is_none());
        assert!(!logged);
    }

    #[test]
    fn eager_eval_poly_gcd_calls_rewrites_call_and_collects_item() {
        let mut ctx = Context::new();
        let a = parse("x^2 + 2*x + 1", &mut ctx).expect("parse a");
        let b = parse("x + 1", &mut ctx).expect("parse b");
        let call = ctx.call("poly_gcd_modp", vec![a, b]);

        let (rewritten, items) =
            eager_eval_poly_gcd_calls_with(&mut ctx, call, true, |_ctx, before, after| {
                (before, after)
            });

        assert!(cas_ast::hold::is_hold(&ctx, rewritten));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, call);
        assert_eq!(items[0].1, rewritten);
    }

    #[test]
    fn eager_eval_poly_gcd_calls_can_skip_item_collection() {
        let mut ctx = Context::new();
        let a = parse("x^2 + 2*x + 1", &mut ctx).expect("parse a");
        let b = parse("x + 1", &mut ctx).expect("parse b");
        let call = ctx.call("pgcdp", vec![a, b]);
        let neg = ctx.add(Expr::Neg(call));

        let (rewritten, items) =
            eager_eval_poly_gcd_calls_with(&mut ctx, neg, false, |_ctx, _, _| ());

        assert!(matches!(ctx.get(rewritten), Expr::Neg(_)));
        assert!(items.is_empty());
    }
}
