// Quadratic Formula Didactic Substeps
//
// Engine wrapper around solver-core didactic quadratic substep construction.

use cas_ast::ExprId;

use crate::engine::Simplifier;
use crate::solver::SolveSubStep;
use crate::step::ImportanceLevel;

/// Apply didactic simplification to all substeps (post-pass).
/// This runs AFTER construction to avoid borrow conflicts.
/// Only applies "safe" simplifications: 0+x->x, 1*x->x, 0*x->0, x/1->x, etc.
pub(crate) fn didactic_simplify_substeps(
    simplifier: &mut Simplifier,
    substeps: &mut [SolveSubStep],
) {
    // Disable step collection during simplification
    let was_collecting = simplifier.collect_steps();
    simplifier.set_collect_steps(false);

    for substep in substeps.iter_mut() {
        // Simplify both LHS and RHS of equation_after
        let (simplified_lhs, _) = simplifier.simplify(substep.equation_after.lhs);
        let (simplified_rhs, _) = simplifier.simplify(substep.equation_after.rhs);

        substep.equation_after.lhs = simplified_lhs;
        substep.equation_after.rhs = simplified_rhs;
    }

    // Restore step collection setting
    simplifier.set_collect_steps(was_collecting);
}

/// Build didactic quadratic substeps using solver-core construction and map
/// to engine step payloads.
pub(crate) fn build_quadratic_substeps(
    simplifier: &mut Simplifier,
    var: &str,
    a: ExprId,
    b: ExprId,
    c: ExprId,
    is_real_only: bool,
) -> Vec<SolveSubStep> {
    let substeps = cas_solver_core::quadratic_didactic::build_quadratic_substeps_with(
        &mut simplifier.context,
        var,
        a,
        b,
        c,
        is_real_only,
        |ctx, id| format!("{}", cas_formatter::DisplayExpr { context: ctx, id }),
    );

    substeps
        .into_iter()
        .map(|substep| SolveSubStep {
            description: substep.description,
            equation_after: substep.equation_after,
            importance: ImportanceLevel::Low,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Simplifier;

    #[test]
    fn test_build_quadratic_substeps() {
        let mut simplifier = Simplifier::with_default_rules();

        // Test with numeric coefficients: x^2 + 2x + 1 = 0
        // General numeric case (c!=0) uses full 7-step derivation
        let a = simplifier.context.num(1);
        let b = simplifier.context.num(2);
        let c = simplifier.context.num(1);

        let steps = build_quadratic_substeps(&mut simplifier, "x", a, b, c, true);

        assert_eq!(
            steps.len(),
            7,
            "General numeric case should generate 7 substeps"
        );
        assert!(steps[0].description.contains("Identificar"));
        assert!(steps[1].description.contains("Dividir"));
        assert!(steps[2].description.contains("Mover"));
        assert!(steps[3].description.contains("Completar"));
        assert!(steps[4].description.contains("cuadrado perfecto"));
        assert!(
            steps[5].description.contains("raiz cuadrada")
                || steps[5].description.contains("raíz cuadrada")
        );
        assert!(steps[6].description.contains("descompone"));
    }
}
