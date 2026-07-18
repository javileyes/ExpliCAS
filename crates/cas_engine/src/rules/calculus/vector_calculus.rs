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
