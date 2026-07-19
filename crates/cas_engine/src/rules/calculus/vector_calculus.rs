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
