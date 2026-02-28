//! Semantic step-analysis helpers shared across runtime crates.

use crate::semantic_equality::SemanticEqualityChecker;
use cas_ast::{Context, ExprId};

/// Returns true when:
/// - no caller-defined didactic step exists, and
/// - final expression is semantically equal to original expression
///   under cycle-check equivalence.
pub fn is_semantic_noop_without_didactic_steps<StepT, FIsDidacticStep>(
    ctx: &Context,
    original_expr: ExprId,
    final_expr: ExprId,
    steps: &[StepT],
    mut is_didactic_step: FIsDidacticStep,
) -> bool
where
    FIsDidacticStep: FnMut(&StepT) -> bool,
{
    if steps.iter().any(&mut is_didactic_step) {
        return false;
    }

    let checker = SemanticEqualityChecker::new(ctx);
    checker.are_equal_for_cycle_check(original_expr, final_expr)
}

#[cfg(test)]
mod tests {
    use super::is_semantic_noop_without_didactic_steps;
    use cas_parser::parse;

    #[derive(Clone)]
    struct StepLite {
        didactic: bool,
    }

    #[test]
    fn returns_true_for_semantic_noop_without_didactic_steps() {
        let mut ctx = cas_ast::Context::new();
        let original = parse("a-b", &mut ctx).expect("parse original");
        let final_expr = parse("a+(-b)", &mut ctx).expect("parse final");
        let steps = vec![StepLite { didactic: false }];

        let out =
            is_semantic_noop_without_didactic_steps(&ctx, original, final_expr, &steps, |s| {
                s.didactic
            });
        assert!(out);
    }

    #[test]
    fn returns_false_when_didactic_steps_exist() {
        let mut ctx = cas_ast::Context::new();
        let original = parse("a-b", &mut ctx).expect("parse original");
        let final_expr = parse("a+(-b)", &mut ctx).expect("parse final");
        let steps = vec![StepLite { didactic: true }];

        let out =
            is_semantic_noop_without_didactic_steps(&ctx, original, final_expr, &steps, |s| {
                s.didactic
            });
        assert!(!out);
    }
}
