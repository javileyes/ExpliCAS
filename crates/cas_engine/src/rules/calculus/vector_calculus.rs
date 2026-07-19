//! Vectorial-calculus verbs over the `Matrix` node (Fase 2, frente vectorial).
//!
//! Every verb shares the same skeleton (V3 debuts it, V4-V6 clone it): the
//! `try_extract_field_vars_call` extractor (arity-2 EXACT, `[vars]` a pure-variable
//! n×1|1×n list — the list-of-vars arity is exclusive to these verbs; `diff`'s
//! 3+-arity SymPy convention is untouched), an assembler in `matrix_rule_support`
//! built on `map_matrix_components`/`differentiate_symbolic_expr`, and a narrated
//! Rewrite. Registration needs TWO cross-crate cables per verb name (the
//! `is_known_eval_engine_function` gate + the rule here) — forgetting either fails
//! SILENTLY (gate-without-rule gotcha).

use crate::define_rule;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::try_extract_field_vars_call;

define_rule!(
    GradientRule,
    "Vector Gradient",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    |ctx, expr| {
        let call = try_extract_field_vars_call(ctx, expr, &["gradient", "grad"])?;
        let result =
            crate::matrix_rule_support::try_gradient_expr(ctx, call.target, &call.var_names)?;
        // Assembling the n×1 column transiently exceeds the anti-worsen node budget
        // (wronskian/matmul bounded-exemption precedent); vars are capped (≤8) in the
        // assembler, output cells = var count. The per-component narration lives in the
        // didactic pipeline (`generate_vector_gradient_substeps`, keyed es/en) — not here.
        Some(
            Rewrite::new(result)
                .desc("Calcular el gradiente del campo escalar")
                .budget_exempt(),
        )
    }
);

define_rule!(
    JacobianRule,
    "Vector Jacobian",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    |ctx, expr| {
        let call = try_extract_field_vars_call(ctx, expr, &["jacobian"])?;
        let result =
            crate::matrix_rule_support::try_jacobian_expr(ctx, call.target, &call.var_names)?;
        // ROWS = functions, COLUMNS = variables (the standard orientation, pinned in
        // fixtures). Caps: ≤8 functions × ≤8 vars = ≤64 cells under the exemption.
        Some(
            Rewrite::new(result)
                .desc("Calcular el jacobiano del campo vectorial")
                .budget_exempt(),
        )
    }
);

define_rule!(
    HessianRule,
    "Vector Hessian",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    |ctx, expr| {
        let call = try_extract_field_vars_call(ctx, expr, &["hessian"])?;
        let result =
            crate::matrix_rule_support::try_hessian_expr(ctx, call.target, &call.var_names)?;
        // Computed directly as the jacobian of the internal gradient (no pipeline
        // re-entry); n×n symmetric for C² fields — the metamorphic fixture ties it to
        // jacobian(gradient(f)) through `equiv`.
        Some(
            Rewrite::new(result)
                .desc("Calcular el hessiano del campo escalar")
                .budget_exempt(),
        )
    }
);

define_rule!(
    DivergenceRule,
    "Vector Divergence",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    |ctx, expr| {
        let call = try_extract_field_vars_call(ctx, expr, &["divergence"])?;
        let result =
            crate::matrix_rule_support::try_divergence_expr(ctx, call.target, &call.var_names)?;
        // Bounded exemption (≤8 component derivatives summed): the anti-worsen budget
        // compares NODES, not shape — a scalar sum of raw quotient derivatives can
        // transiently exceed it (laplacian(ln(x²+y²)) was a false residual without it).
        Some(
            Rewrite::new(result)
                .desc("Calcular la divergencia del campo vectorial")
                .budget_exempt(),
        )
    }
);

define_rule!(
    LaplacianRule,
    "Vector Laplacian",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    |ctx, expr| {
        let call = try_extract_field_vars_call(ctx, expr, &["laplacian"])?;
        let result =
            crate::matrix_rule_support::try_laplacian_expr(ctx, call.target, &call.var_names)?;
        // Δf = Σ ∂²f/∂vᵢ² computed internally (div ∘ grad without pipeline re-entry);
        // vector-laplacian stays a named scope-out (Matrix target declines). Bounded
        // exemption: ≤8 second derivatives summed (see DivergenceRule note).
        Some(
            Rewrite::new(result)
                .desc("Calcular el laplaciano del campo escalar")
                .budget_exempt(),
        )
    }
);

define_rule!(
    CurlRule,
    "Vector Curl",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    |ctx, expr| {
        let call = try_extract_field_vars_call(ctx, expr, &["curl", "rot"])?;
        let result = crate::matrix_rule_support::try_curl_expr(ctx, call.target, &call.var_names)?;
        // 3D → 3×1 column; 2D → SCALAR ∂Q/∂x − ∂P/∂y (never zero-padded). Bounded
        // exemption: ≤6 raw derivative differences assembled (see DivergenceRule note —
        // the anti-worsen budget counts nodes, not shapes).
        Some(
            Rewrite::new(result)
                .desc("Calcular el rotacional del campo vectorial")
                .budget_exempt(),
        )
    }
);

/// Positional extraction for `lineintegral(field, [vars], [param], t, a, b)`
/// (F4, Fase 3 — arity-6 EXACT; the arity-2 field+[vars] extractor of the
/// other verbs does not fit this shape). Validates: `[vars]` a pure-Variable
/// n×1|1×n list; `[param]` a Matrix of the SAME length whose entries mention
/// NEITHER the field variables NOR each other's aliases (they are functions of
/// the parameter); `t` a Variable that is not one of the field variables.
struct LineIntegralCall {
    field: cas_ast::ExprId,
    var_entries: Vec<(cas_ast::ExprId, cas_ast::symbol::SymbolId)>,
    param_comps: Vec<cas_ast::ExprId>,
    t_expr: cas_ast::ExprId,
    lower: cas_ast::ExprId,
    upper: cas_ast::ExprId,
}

fn try_extract_lineintegral_call(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<LineIntegralCall> {
    if !ctx.is_call_named(expr, "lineintegral") {
        return None;
    }
    let cas_ast::Expr::Function(_, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 6 {
        return None;
    }
    let args = args.clone();
    let (field, vars_expr, param_expr, t_expr, lower, upper) =
        (args[0], args[1], args[2], args[3], args[4], args[5]);
    let cas_ast::Expr::Matrix { rows, cols, data } = ctx.get(vars_expr) else {
        return None;
    };
    if (*rows != 1 && *cols != 1) || data.is_empty() {
        return None;
    }
    let mut var_entries = Vec::with_capacity(data.len());
    for &v in data {
        let cas_ast::Expr::Variable(sym) = ctx.get(v) else {
            return None;
        };
        var_entries.push((v, *sym));
    }
    let cas_ast::Expr::Variable(t_sym) = ctx.get(t_expr) else {
        return None;
    };
    if var_entries.iter().any(|(_, sym)| *sym == *t_sym) {
        return None;
    }
    let cas_ast::Expr::Matrix { rows, cols, data } = ctx.get(param_expr) else {
        return None;
    };
    if (*rows != 1 && *cols != 1) || data.len() != var_entries.len() {
        return None;
    }
    let param_comps = data.clone();
    for &comp in &param_comps {
        if var_entries
            .iter()
            .any(|(_, sym)| expr_mentions_symbol(ctx, comp, *sym))
        {
            return None;
        }
    }
    Some(LineIntegralCall {
        field,
        var_entries,
        param_comps,
        t_expr,
        lower,
        upper,
    })
}

/// True when `expr` mentions the variable symbol anywhere.
fn expr_mentions_symbol(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    var_sym: cas_ast::symbol::SymbolId,
) -> bool {
    use cas_ast::Expr;
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Variable(sym) if *sym == var_sym => return true,
            Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
        }
    }
    false
}

/// Substitute the parametrization into an expression: every field variable is
/// replaced by its parametrization component. Sequential substitution is
/// order-safe because the extractor guarantees no component mentions a field
/// variable.
fn substitute_parametrization(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    var_entries: &[(cas_ast::ExprId, cas_ast::symbol::SymbolId)],
    param_comps: &[cas_ast::ExprId],
) -> cas_ast::ExprId {
    let mut result = expr;
    for ((var_expr, _), comp) in var_entries.iter().zip(param_comps) {
        result = cas_ast::traversal::substitute_expr_by_id(ctx, result, *var_expr, *comp);
    }
    result
}

define_rule!(
    LineIntegralRule,
    "Line Integral",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    |ctx, expr| {
        // F4 (Fase 3): pure ASSEMBLER over live composition — parametrize,
        // differentiate, build the integrand, and hand the DEFINITE integral to
        // the integration engine (clean value when elementary, honest residual
        // otherwise). All-or-nothing: any non-differentiable component declines
        // the whole call.
        let call = try_extract_lineintegral_call(ctx, expr)?;
        let t_name = match ctx.get(call.t_expr) {
            cas_ast::Expr::Variable(sym) => ctx.sym_name(*sym).to_string(),
            _ => return None,
        };
        let mut derivatives = Vec::with_capacity(call.param_comps.len());
        for &comp in &call.param_comps {
            derivatives.push(
                cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
                    ctx, comp, &t_name,
                )?,
            );
        }
        let integrand = match ctx.get(call.field).clone() {
            // Vector field: ∮ F·dr = ∫ Σ Fᵢ(r(t))·rᵢ'(t) dt — requires
            // #components == #vars.
            cas_ast::Expr::Matrix { rows, cols, data } => {
                if (rows != 1 && cols != 1) || data.len() != call.var_entries.len() {
                    return None;
                }
                let mut sum: Option<cas_ast::ExprId> = None;
                for (&f_i, &dr_i) in data.iter().zip(&derivatives) {
                    let substituted =
                        substitute_parametrization(ctx, f_i, &call.var_entries, &call.param_comps);
                    let term = ctx.add(cas_ast::Expr::Mul(substituted, dr_i));
                    sum = Some(match sum {
                        None => term,
                        Some(acc) => ctx.add(cas_ast::Expr::Add(acc, term)),
                    });
                }
                sum?
            }
            // Scalar field: ∫ f ds = ∫ f(r(t))·‖r'(t)‖ dt.
            _ => {
                let substituted = substitute_parametrization(
                    ctx,
                    call.field,
                    &call.var_entries,
                    &call.param_comps,
                );
                let mut norm_sq: Option<cas_ast::ExprId> = None;
                for &dr_i in &derivatives {
                    let two = ctx.num(2);
                    let sq = ctx.add(cas_ast::Expr::Pow(dr_i, two));
                    norm_sq = Some(match norm_sq {
                        None => sq,
                        Some(acc) => ctx.add(cas_ast::Expr::Add(acc, sq)),
                    });
                }
                let speed = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![norm_sq?]);
                ctx.add(cas_ast::Expr::Mul(substituted, speed))
            }
        };
        let result = ctx.call(
            "integrate",
            vec![integrand, call.t_expr, call.lower, call.upper],
        );
        Some(
            Rewrite::new(result)
                .desc("Integral de línea: parametrizar, derivar y ensamblar el integrando")
                .budget_exempt(),
        )
    }
);

/// Positional extraction for
/// `surface_integral(field, [vars], r, [u,v], [a,b], [c,d])` (F5, Fase 3 —
/// arity-6 EXACT, 3D surfaces only: the area element needs the 3-component
/// cross product). Validates like the line-integral extractor: `[vars]` three
/// pure Variables; `r` a 3-component Matrix free of the field variables;
/// `[u,v]` two DISTINCT parameter Variables outside the field variables; the
/// two ranges 2-entry Matrices free of the field variables (the inner range
/// may mention the outer parameter — iterated integrals compose).
struct SurfaceIntegralCall {
    field: cas_ast::ExprId,
    var_entries: Vec<(cas_ast::ExprId, cas_ast::symbol::SymbolId)>,
    r_comps: Vec<cas_ast::ExprId>,
    u_expr: cas_ast::ExprId,
    v_expr: cas_ast::ExprId,
    u_range: (cas_ast::ExprId, cas_ast::ExprId),
    v_range: (cas_ast::ExprId, cas_ast::ExprId),
}

fn try_extract_surface_integral_call(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<SurfaceIntegralCall> {
    if !ctx.is_call_named(expr, "surface_integral") {
        return None;
    }
    let cas_ast::Expr::Function(_, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 6 {
        return None;
    }
    let args = args.clone();
    let (field, vars_expr, r_expr, params_expr, u_range_expr, v_range_expr) =
        (args[0], args[1], args[2], args[3], args[4], args[5]);
    let cas_ast::Expr::Matrix { rows, cols, data } = ctx.get(vars_expr) else {
        return None;
    };
    if (*rows != 1 && *cols != 1) || data.len() != 3 {
        return None;
    }
    let mut var_entries = Vec::with_capacity(3);
    for &v in data {
        let cas_ast::Expr::Variable(sym) = ctx.get(v) else {
            return None;
        };
        var_entries.push((v, *sym));
    }
    let cas_ast::Expr::Matrix { rows, cols, data } = ctx.get(params_expr) else {
        return None;
    };
    if (*rows != 1 && *cols != 1) || data.len() != 2 {
        return None;
    }
    let (u_expr, v_expr) = (data[0], data[1]);
    let (cas_ast::Expr::Variable(u_sym), cas_ast::Expr::Variable(v_sym)) =
        (ctx.get(u_expr), ctx.get(v_expr))
    else {
        return None;
    };
    let (u_sym, v_sym) = (*u_sym, *v_sym);
    if u_sym == v_sym || var_entries.iter().any(|(_, s)| *s == u_sym || *s == v_sym) {
        return None;
    }
    let cas_ast::Expr::Matrix { rows, cols, data } = ctx.get(r_expr) else {
        return None;
    };
    if (*rows != 1 && *cols != 1) || data.len() != 3 {
        return None;
    }
    let r_comps = data.clone();
    let range_of = |ctx: &cas_ast::Context,
                    e: cas_ast::ExprId|
     -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
        let cas_ast::Expr::Matrix { rows, cols, data } = ctx.get(e) else {
            return None;
        };
        if (*rows != 1 && *cols != 1) || data.len() != 2 {
            return None;
        }
        Some((data[0], data[1]))
    };
    let u_range = range_of(ctx, u_range_expr)?;
    let v_range = range_of(ctx, v_range_expr)?;
    for &e in r_comps
        .iter()
        .chain([u_range.0, u_range.1, v_range.0, v_range.1].iter())
    {
        if var_entries
            .iter()
            .any(|(_, sym)| expr_mentions_symbol(ctx, e, *sym))
        {
            return None;
        }
    }
    Some(SurfaceIntegralCall {
        field,
        var_entries,
        r_comps,
        u_expr,
        v_expr,
        u_range,
        v_range,
    })
}

define_rule!(
    SurfaceIntegralRule,
    "Surface Integral",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    |ctx, expr| {
        // F5 (Fase 3): assembler over live composition — differentiate the
        // parametrization, build the area element `r_u × r_v`, assemble
        // `f(r)·‖r_u×r_v‖` (scalar) or `F(r)·(r_u×r_v)` (vector flux), and hand
        // the ITERATED definite integral to the integration engine. A
        // non-elementary integrand stays an honest residual integral
        // (paraboloid pin) — the verb never forces a value.
        let call = try_extract_surface_integral_call(ctx, expr)?;
        let u_name = match ctx.get(call.u_expr) {
            cas_ast::Expr::Variable(sym) => ctx.sym_name(*sym).to_string(),
            _ => return None,
        };
        let v_name = match ctx.get(call.v_expr) {
            cas_ast::Expr::Variable(sym) => ctx.sym_name(*sym).to_string(),
            _ => return None,
        };
        let mut r_u = Vec::with_capacity(3);
        let mut r_v = Vec::with_capacity(3);
        for &comp in &call.r_comps {
            r_u.push(
                cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
                    ctx, comp, &u_name,
                )?,
            );
            r_v.push(
                cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
                    ctx, comp, &v_name,
                )?,
            );
        }
        let mk = |data: Vec<cas_ast::ExprId>| cas_math::matrix::Matrix {
            rows: 3,
            cols: 1,
            data,
        };
        let (mu, mv) = (mk(r_u), mk(r_v));
        let cross_expr = crate::matrix_rule_support::matrix_cross(ctx, &mu, &mv)?;
        let cas_ast::Expr::Matrix { data: cross, .. } = ctx.get(cross_expr).clone() else {
            return None;
        };
        let integrand = match ctx.get(call.field).clone() {
            // Vector flux: ∫∫ F·(r_u×r_v) dv du — requires 3 components.
            cas_ast::Expr::Matrix { rows, cols, data } => {
                if (rows != 1 && cols != 1) || data.len() != 3 {
                    return None;
                }
                let mut sum: Option<cas_ast::ExprId> = None;
                for (&f_i, &c_i) in data.iter().zip(&cross) {
                    let substituted =
                        substitute_parametrization(ctx, f_i, &call.var_entries, &call.r_comps);
                    let term = ctx.add(cas_ast::Expr::Mul(substituted, c_i));
                    sum = Some(match sum {
                        None => term,
                        Some(acc) => ctx.add(cas_ast::Expr::Add(acc, term)),
                    });
                }
                sum?
            }
            // Scalar: ∫∫ f(r)·‖r_u×r_v‖ dv du.
            _ => {
                let substituted =
                    substitute_parametrization(ctx, call.field, &call.var_entries, &call.r_comps);
                let mut norm_sq: Option<cas_ast::ExprId> = None;
                for &c_i in &cross {
                    let two = ctx.num(2);
                    let sq = ctx.add(cas_ast::Expr::Pow(c_i, two));
                    norm_sq = Some(match norm_sq {
                        None => sq,
                        Some(acc) => ctx.add(cas_ast::Expr::Add(acc, sq)),
                    });
                }
                let area_element = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![norm_sq?]);
                ctx.add(cas_ast::Expr::Mul(substituted, area_element))
            }
        };
        let inner = ctx.call(
            "integrate",
            vec![integrand, call.v_expr, call.v_range.0, call.v_range.1],
        );
        let result = ctx.call(
            "integrate",
            vec![inner, call.u_expr, call.u_range.0, call.u_range.1],
        );
        Some(
            Rewrite::new(result)
                .desc(
                    "Integral de superficie: parametrizar, ensamblar el elemento de área e iterar",
                )
                .budget_exempt(),
        )
    }
);

/// Canonical polynomial rebuild (MultiPoly round-trip): collapses raw
/// derivative litter like `x² − x²·1` to its normal form. `None` when the
/// expression is not polynomial under the comparator's budget.
fn canonicalize_poly(ctx: &mut cas_ast::Context, expr: cas_ast::ExprId) -> Option<cas_ast::ExprId> {
    let budget = cas_math::multipoly::PolyBudget::default();
    let poly = cas_math::multipoly::conversion::multipoly_from_expr(ctx, expr, &budget).ok()?;
    Some(cas_math::multipoly::conversion::multipoly_to_expr(
        &poly, ctx,
    ))
}

/// The interim-path reconstruction + exact verification behind
/// [`PotentialRule`], split out for unit testing. `None` = honest decline.
pub(crate) fn try_potential_expr(
    ctx: &mut cas_ast::Context,
    comps: &[cas_ast::ExprId],
    var_names: &[String],
) -> Option<cas_ast::ExprId> {
    let clean_integral = |ctx: &mut cas_ast::Context,
                          target: cas_ast::ExprId,
                          var: &str|
     -> Option<cas_ast::ExprId> {
        let outcome = super::integration::integrate_with_trace(ctx, target, var)?;
        if !outcome.required_conditions.is_empty() {
            return None;
        }
        Some(outcome.result)
    };
    let mut candidate = clean_integral(ctx, comps[0], &var_names[0])?;
    for k in 1..comps.len() {
        let partial = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
            ctx,
            candidate,
            &var_names[k],
        )?;
        let rest = ctx.add(cas_ast::Expr::Sub(comps[k], partial));
        // The raw difference carries unsimplified litter (`x² − x²·1`) that
        // stalls the integrator — canonicalize through MultiPoly when the
        // field is polynomial (the verifier's normal form); a non-polynomial
        // rest integrates raw or declines honest.
        let rest = canonicalize_poly(ctx, rest).unwrap_or(rest);
        let piece = clean_integral(ctx, rest, &var_names[k])?;
        candidate = ctx.add(cas_ast::Expr::Add(candidate, piece));
    }
    // Exact verification: ∂φ/∂xᵢ ≡ Fᵢ for EVERY component, or decline. The raw
    // derivative carries non-literal exponents (`x^(2-1)`) that the polynomial
    // converter rejects — fold constants first (the recorded lesson).
    for (comp, var) in comps.iter().zip(var_names) {
        let derived = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
            ctx, candidate, var,
        )?;
        let derived = cas_math::limits_support::fold_constant_subexprs(ctx, derived);
        let comp_folded = cas_math::limits_support::fold_constant_subexprs(ctx, *comp);
        if !cas_math::poly_compare::poly_eq(ctx, derived, comp_folded) {
            return None;
        }
    }
    Some(candidate)
}

define_rule!(
    PotentialRule,
    "Scalar Potential",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    |ctx, expr| {
        // F6 (Fase 3): scalar potential of a conservative field by the interim
        // path — φ = ∫F₁ dx₁, then for each further variable integrate the
        // difference F_k − ∂φ/∂x_k. EMISSION IS GATED BY VERIFICATION, not by
        // the construction: every component must satisfy `∂φ/∂xᵢ ≡ Fᵢ` under
        // the EXACT polynomial comparator (`poly_eq` — conversion failure
        // declines conservatively). A non-conservative field can never verify,
        // so it declines to the honest residual; the additive constant is
        // implicit (documented in the examples row).
        let call = try_extract_field_vars_call(ctx, expr, &["potential"])?;
        let cas_ast::Expr::Matrix { rows, cols, data } = ctx.get(call.target) else {
            return None;
        };
        if (*rows != 1 && *cols != 1)
            || data.len() != call.var_names.len()
            || !(2..=3).contains(&data.len())
        {
            return None;
        }
        let comps = data.clone();
        let var_names = call.var_names.clone();
        let candidate = try_potential_expr(ctx, &comps, &var_names)?;
        Some(
            Rewrite::new(candidate)
                .desc("Potencial escalar: reconstruir por caminos y verificar \u{2207}\u{3c6} = F")
                .budget_exempt(),
        )
    }
);

#[cfg(test)]
mod potential_tests {
    use super::*;

    #[test]
    fn potential_reconstructs_polynomial_fields() {
        let mut ctx = cas_ast::Context::new();
        let f1 = cas_parser::parse("2*x*y", &mut ctx).expect("f1");
        let f2 = cas_parser::parse("x^2", &mut ctx).expect("f2");
        let vars = vec!["x".to_string(), "y".to_string()];
        let got = try_potential_expr(&mut ctx, &[f1, f2], &vars);
        match got {
            Some(id) => {
                let s = format!("{}", cas_formatter::DisplayExpr { context: &ctx, id });
                assert!(s.contains('x') && s.contains('y'), "{s}");
            }
            None => panic!("debe reconstruir el potencial de [2xy, x^2]"),
        }
    }
}

define_rule!(
    LimitMultivarRule,
    "Multivariate Limit",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    |ctx, expr, parent_ctx| {
        // F7 (Fase 3): `limit(f, [x,y], [a,b])` by PROVEN continuity only —
        // polynomial/rational trees with the substituted denominator folding
        // to a nonzero exact rational, plus continuous total-real unary calls.
        // Everything else stays an honest residual echo: multivariate limit
        // existence is path-dependent and F8 owns the negative side. The rule
        // is VALUE-DEPENDENT (guardrail #1): the continuity reasoning is
        // real-order, so the complex domain declines (the F0 kill-switch
        // discipline, inherited).
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::ComplexEnabled {
            return None;
        }
        if !ctx.is_call_named(expr, "limit") {
            return None;
        }
        let cas_ast::Expr::Function(_, args) = ctx.get(expr) else {
            return None;
        };
        if args.len() != 3 {
            return None;
        }
        let args = args.clone();
        let (target, vars_expr, points_expr) = (args[0], args[1], args[2]);
        // List form ONLY — the univariate `limit(f, x, a)` in expression
        // position stays an honest residual echo (F9 evaluates it).
        let cas_ast::Expr::Matrix { rows, cols, data } = ctx.get(vars_expr) else {
            return None;
        };
        if (*rows != 1 && *cols != 1) || data.is_empty() {
            return None;
        }
        let mut var_ids = Vec::with_capacity(data.len());
        let mut var_syms = Vec::with_capacity(data.len());
        for &v in data {
            let cas_ast::Expr::Variable(sym) = ctx.get(v) else {
                return None;
            };
            var_ids.push(v);
            var_syms.push(*sym);
        }
        let cas_ast::Expr::Matrix { rows, cols, data } = ctx.get(points_expr) else {
            return None;
        };
        if (*rows != 1 && *cols != 1) || data.len() != var_ids.len() {
            return None;
        }
        let points = data.clone();
        // Point entries: free of the expansion variables AND finite (an
        // at-infinity multivariate limit is out of scope — named residual);
        // imaginary points inherit the F0 discipline via the same detector.
        for &pt in &points {
            if var_syms
                .iter()
                .any(|sym| expr_mentions_symbol(ctx, pt, *sym))
            {
                return None;
            }
            if expr_contains_nonfinite_constant(ctx, pt)
                || cas_math::numeric_eval::expr_contains_imaginary(ctx, pt)
            {
                return None;
            }
        }
        if let Some(value) = cas_math::limits_support::try_multivar_limit_by_continuity(
            ctx, target, &var_ids, &points,
        ) {
            return Some(
                Rewrite::new(value)
                    .desc("Límite multivariable por continuidad probada: sustituir el punto")
                    .budget_exempt(),
            );
        }
        // F8: continuity declined — run the path battery. The verdict APPLIES
        // a rewrite (→ undefined), so the witnesses travel as an assumption
        // event; agreeing paths NEVER prove existence (the residual echo stays).
        let verdict =
            cas_math::limits_support::try_multivar_dne_by_paths(ctx, target, &var_ids, &points)?;
        let message = match &verdict.witness_b {
            Some(b) => format!(
                "el límite no existe: por {} el límite es {}; por {} es {}",
                verdict.witness_a.path_display,
                verdict.witness_a.value_display,
                b.path_display,
                b.value_display
            ),
            None => format!(
                "el límite no existe: por {} {}",
                verdict.witness_a.path_display, verdict.witness_a.value_display
            ),
        };
        let undefined = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::Undefined));
        let event = crate::AssumptionEvent {
            key: cas_solver_core::assumption_model::AssumptionKey::Defined {
                expr_fingerprint: cas_solver_core::assumption_model::expr_fingerprint(ctx, target),
            },
            expr_display: message.clone(),
            message,
            // HeuristicAssumption es el único kind que collect_domain_warnings
            // deja pasar — el mensaje es un VEREDICTO informativo, no una
            // asunción; el canal default-mode para veredictos de rules es el
            // residual transversal nombrado (F1/F7/F8 lo esperan).
            kind: cas_solver_core::assumption_model::AssumptionKind::HeuristicAssumption,
            expr_id: Some(target),
        };
        Some(
            Rewrite::new(undefined)
                .desc("Límite multivariable: caminos con límites distintos — no existe")
                .assume(event)
                .budget_exempt(),
        )
    }
);

/// True when `expr` contains an `Infinity`/`Undefined` sentinel anywhere.
fn expr_contains_nonfinite_constant(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    use cas_ast::Expr;
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Constant(cas_ast::Constant::Infinity | cas_ast::Constant::Undefined) => {
                return true
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            _ => {}
        }
    }
    false
}

define_rule!(
    LimitUnivarExprRule,
    "Nested Limit",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    |ctx, expr, parent_ctx| {
        // F9 (Fase 3): `limit(f, x, a)` in EXPRESSION position (nested,
        // composed, iterated) evaluates through the univariate limit engine
        // and emits ONLY on a COMPLETE resolution — a residual, a warning, or
        // an inner unevaluated calculus call decline all-or-nothing (a
        // residual is never operated on as if it were a value). Iterated
        // limits resolve bottom-up: the inner call becomes a value (possibly
        // parametric) before the outer fires, so nesting depth is naturally
        // bounded by the clean-target guard. Value-dependent (guardrail #1):
        // the engine reasons with the real order — complex declines (F0
        // discipline; F11 re-grants selectively).
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::ComplexEnabled {
            return None;
        }
        if !ctx.is_call_named(expr, "limit") {
            return None;
        }
        let cas_ast::Expr::Function(_, args) = ctx.get(expr) else {
            return None;
        };
        if args.len() != 3 {
            return None;
        }
        let args = args.clone();
        let (target, var_expr, point) = (args[0], args[1], args[2]);
        // Univar shape ONLY (the list form is LimitMultivarRule's).
        if !matches!(ctx.get(var_expr), cas_ast::Expr::Variable(_)) {
            return None;
        }
        if matches!(ctx.get(point), cas_ast::Expr::Matrix { .. }) {
            return None;
        }
        // All-or-nothing: an unresolved calculus call inside target or point
        // would be misread as an opaque constant by the limit rules.
        if crate::rules::functions::contains_unevaluated_calculus_call(ctx, target)
            || crate::rules::functions::contains_unevaluated_calculus_call(ctx, point)
        {
            return None;
        }
        let approach = match ctx.get(point) {
            cas_ast::Expr::Constant(cas_ast::Constant::Infinity) => {
                cas_math::limit_types::Approach::PosInfinity
            }
            cas_ast::Expr::Neg(inner)
                if matches!(
                    ctx.get(*inner),
                    cas_ast::Expr::Constant(cas_ast::Constant::Infinity)
                ) =>
            {
                cas_math::limit_types::Approach::NegInfinity
            }
            _ => cas_math::limit_types::Approach::Finite(point),
        };
        let opts = cas_math::limit_types::LimitOptions::default();
        let outcome = cas_math::limits_support::eval_limit_at_infinity(
            ctx, target, var_expr, approach, &opts,
        );
        // COMPLETE resolutions only: any warning (residual, proven-DNE with
        // its citation, imaginary point) declines to the honest echo — the
        // action path owns warnings; a rule cannot carry them on decline.
        if outcome.warning.is_some() {
            return None;
        }
        if crate::rules::functions::contains_unevaluated_calculus_call(ctx, outcome.expr) {
            return None;
        }
        Some(
            Rewrite::new(outcome.expr)
                .desc("Evaluar el límite anidado (límite iterado, no doble)")
                .budget_exempt(),
        )
    }
);
