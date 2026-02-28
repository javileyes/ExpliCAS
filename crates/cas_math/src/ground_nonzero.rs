//! Generic ground-expression non-zero fallback pipeline.

use crate::ground_eval_guard::GroundEvalGuard;
use cas_ast::{Context, ExprId};

/// Attempt to prove non-zero for a ground expression via three phases:
/// 1) evaluate expression in a caller-provided evaluator context,
/// 2) classify direct numeric result,
/// 3) run a structural fallback on the evaluated expression.
///
/// Returns `None` when:
/// - re-entrancy guard blocks execution,
/// - evaluator declines (`None`),
/// - both numeric and structural checks are inconclusive.
pub fn try_ground_nonzero_with<ProofT, FEvaluate, FClassifyNumeric, FStructuralFallback>(
    ctx: &Context,
    expr: ExprId,
    mut evaluate: FEvaluate,
    mut classify_numeric: FClassifyNumeric,
    mut structural_fallback: FStructuralFallback,
) -> Option<ProofT>
where
    FEvaluate: FnMut(&Context, ExprId) -> Option<(Context, ExprId)>,
    FClassifyNumeric: FnMut(&Context, ExprId) -> Option<ProofT>,
    FStructuralFallback: FnMut(&Context, ExprId) -> Option<ProofT>,
{
    let _guard = GroundEvalGuard::enter()?;
    let (evaluated_ctx, evaluated_expr) = evaluate(ctx, expr)?;

    if let Some(proof) = classify_numeric(&evaluated_ctx, evaluated_expr) {
        return Some(proof);
    }

    structural_fallback(&evaluated_ctx, evaluated_expr)
}

#[cfg(test)]
mod tests {
    use super::try_ground_nonzero_with;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Proof {
        Proven,
        Disproven,
    }

    #[test]
    fn returns_numeric_classification_when_available() {
        let mut ctx = cas_ast::Context::new();
        let one = ctx.num(1);
        let out = try_ground_nonzero_with(
            &ctx,
            one,
            |src, id| Some((src.clone(), id)),
            |_src, _id| Some(Proof::Proven),
            |_src, _id| None,
        );
        assert_eq!(out, Some(Proof::Proven));
    }

    #[test]
    fn falls_back_to_structural_when_numeric_not_available() {
        let mut ctx = cas_ast::Context::new();
        let one = ctx.num(1);
        let out = try_ground_nonzero_with(
            &ctx,
            one,
            |src, id| Some((src.clone(), id)),
            |_src, _id| None,
            |_src, _id| Some(Proof::Disproven),
        );
        assert_eq!(out, Some(Proof::Disproven));
    }
}
