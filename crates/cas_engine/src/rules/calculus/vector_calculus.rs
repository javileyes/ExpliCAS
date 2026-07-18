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
