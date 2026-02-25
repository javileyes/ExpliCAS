use crate::verification::VerifyStatus;
use cas_ast::{Equation, ExprId};

/// Verify one candidate solution by substitution and two-phase simplification
/// using caller-provided hooks.
#[allow(clippy::too_many_arguments)]
pub fn verify_solution_with<
    FSubstituteDiff,
    FSimplifyStrict,
    FSimplifyGeneric,
    FFoldNumericIslands,
    FIsNumericZero,
    FContainsVariable,
    FRenderExpr,
>(
    equation: &Equation,
    var: &str,
    solution: ExprId,
    mut substitute_diff: FSubstituteDiff,
    mut simplify_strict: FSimplifyStrict,
    mut simplify_generic: FSimplifyGeneric,
    mut fold_numeric_islands: FFoldNumericIslands,
    mut is_numeric_zero: FIsNumericZero,
    mut contains_variable: FContainsVariable,
    mut render_expr: FRenderExpr,
) -> VerifyStatus
where
    FSubstituteDiff: FnMut(&Equation, &str, ExprId) -> ExprId,
    FSimplifyStrict: FnMut(ExprId) -> ExprId,
    FSimplifyGeneric: FnMut(ExprId) -> ExprId,
    FFoldNumericIslands: FnMut(ExprId) -> ExprId,
    FIsNumericZero: FnMut(ExprId) -> bool,
    FContainsVariable: FnMut(ExprId) -> bool,
    FRenderExpr: FnMut(ExprId) -> String,
{
    let diff = substitute_diff(equation, var, solution);

    let strict_result = simplify_strict(diff);
    if is_numeric_zero(strict_result) {
        return VerifyStatus::Verified;
    }

    if contains_variable(strict_result) {
        crate::verify_stats::record_attempted();
        let folded = fold_numeric_islands(strict_result);
        if folded != strict_result {
            crate::verify_stats::record_changed();
            let folded_result = simplify_strict(folded);
            if is_numeric_zero(folded_result) {
                crate::verify_stats::record_verified();
                return VerifyStatus::Verified;
            }
        }
    }

    if !contains_variable(strict_result) {
        let generic_result = simplify_generic(diff);
        if is_numeric_zero(generic_result) {
            return VerifyStatus::Verified;
        }
    }

    VerifyStatus::Unverifiable {
        residual: strict_result,
        reason: format!("residual: {}", render_expr(strict_result)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify_substitution::substitute_equation_diff;
    use cas_ast::{Context, Equation, RelOp};

    fn sample_equation(ctx: &mut Context) -> (Equation, ExprId) {
        let x = ctx.var("x");
        let one = ctx.num(1);
        (
            Equation {
                lhs: x,
                rhs: one,
                op: RelOp::Eq,
            },
            one,
        )
    }

    #[test]
    fn verify_solution_with_accepts_strict_zero() {
        let mut context = Context::new();
        let zero = context.num(0);
        let one = context.num(1);
        let (equation, solution) = sample_equation(&mut context);
        let context_cell = std::cell::RefCell::new(context);
        let strict_results = std::cell::RefCell::new(vec![zero]);
        let strict_calls = std::cell::Cell::new(0usize);
        let generic_calls = std::cell::Cell::new(0usize);
        let fold_calls = std::cell::Cell::new(0usize);

        let out = verify_solution_with(
            &equation,
            "x",
            solution,
            |eq, name, sol| {
                let mut ctx = context_cell.borrow_mut();
                substitute_equation_diff(&mut ctx, eq, name, sol)
            },
            |_expr| {
                strict_calls.set(strict_calls.get() + 1);
                strict_results.borrow_mut().remove(0)
            },
            |_expr| {
                generic_calls.set(generic_calls.get() + 1);
                one
            },
            |_expr| {
                fold_calls.set(fold_calls.get() + 1);
                one
            },
            |expr| expr == zero,
            |_expr| false,
            |expr| format!("expr#{:?}", expr),
        );

        assert!(matches!(out, VerifyStatus::Verified));
        assert_eq!(strict_calls.get(), 1);
        assert_eq!(generic_calls.get(), 0);
        assert_eq!(fold_calls.get(), 0);
    }

    #[test]
    fn verify_solution_with_accepts_after_phase_15_fold() {
        let mut context = Context::new();
        let x = context.var("x");
        let one = context.num(1);
        let zero = context.num(0);
        let folded = context.num(2);
        let (equation, solution) = sample_equation(&mut context);
        let context_cell = std::cell::RefCell::new(context);
        let strict_results = std::cell::RefCell::new(vec![x, zero]);
        let strict_calls = std::cell::Cell::new(0usize);
        let generic_calls = std::cell::Cell::new(0usize);
        let fold_calls = std::cell::Cell::new(0usize);

        let out = verify_solution_with(
            &equation,
            "x",
            solution,
            |eq, name, sol| {
                let mut ctx = context_cell.borrow_mut();
                substitute_equation_diff(&mut ctx, eq, name, sol)
            },
            |_expr| {
                strict_calls.set(strict_calls.get() + 1);
                strict_results.borrow_mut().remove(0)
            },
            |_expr| {
                generic_calls.set(generic_calls.get() + 1);
                one
            },
            |_expr| {
                fold_calls.set(fold_calls.get() + 1);
                folded
            },
            |expr| expr == zero,
            |expr| expr == x,
            |expr| format!("expr#{:?}", expr),
        );

        assert!(matches!(out, VerifyStatus::Verified));
        assert_eq!(strict_calls.get(), 2);
        assert_eq!(generic_calls.get(), 0);
        assert_eq!(fold_calls.get(), 1);
    }

    #[test]
    fn verify_solution_with_accepts_after_generic_fallback() {
        let mut context = Context::new();
        let one = context.num(1);
        let two = context.num(2);
        let zero = context.num(0);
        let (equation, solution) = sample_equation(&mut context);
        let context_cell = std::cell::RefCell::new(context);
        let strict_results = std::cell::RefCell::new(vec![two]);
        let strict_calls = std::cell::Cell::new(0usize);
        let generic_calls = std::cell::Cell::new(0usize);
        let fold_calls = std::cell::Cell::new(0usize);

        let out = verify_solution_with(
            &equation,
            "x",
            solution,
            |eq, name, sol| {
                let mut ctx = context_cell.borrow_mut();
                substitute_equation_diff(&mut ctx, eq, name, sol)
            },
            |_expr| {
                strict_calls.set(strict_calls.get() + 1);
                strict_results.borrow_mut().remove(0)
            },
            |_expr| {
                generic_calls.set(generic_calls.get() + 1);
                zero
            },
            |_expr| {
                fold_calls.set(fold_calls.get() + 1);
                one
            },
            |expr| expr == zero,
            |_expr| false,
            |expr| format!("expr#{:?}", expr),
        );

        assert!(matches!(out, VerifyStatus::Verified));
        assert_eq!(strict_calls.get(), 1);
        assert_eq!(generic_calls.get(), 1);
        assert_eq!(fold_calls.get(), 0);
    }

    #[test]
    fn verify_solution_with_reports_unverifiable_when_checks_fail() {
        let mut context = Context::new();
        let one = context.num(1);
        let two = context.num(2);
        let three = context.num(3);
        let zero = context.num(0);
        let (equation, solution) = sample_equation(&mut context);
        let context_cell = std::cell::RefCell::new(context);
        let strict_results = std::cell::RefCell::new(vec![two]);
        let strict_calls = std::cell::Cell::new(0usize);
        let generic_calls = std::cell::Cell::new(0usize);
        let fold_calls = std::cell::Cell::new(0usize);

        let out = verify_solution_with(
            &equation,
            "x",
            solution,
            |eq, name, sol| {
                let mut ctx = context_cell.borrow_mut();
                substitute_equation_diff(&mut ctx, eq, name, sol)
            },
            |_expr| {
                strict_calls.set(strict_calls.get() + 1);
                strict_results.borrow_mut().remove(0)
            },
            |_expr| {
                generic_calls.set(generic_calls.get() + 1);
                three
            },
            |_expr| {
                fold_calls.set(fold_calls.get() + 1);
                one
            },
            |expr| expr == zero,
            |_expr| false,
            |expr| format!("expr#{:?}", expr),
        );

        match out {
            VerifyStatus::Unverifiable { residual, reason } => {
                assert_eq!(residual, two);
                assert!(reason.contains("residual: expr#"));
            }
            other => panic!("expected unverifiable, got {:?}", other),
        }
        assert_eq!(strict_calls.get(), 1);
        assert_eq!(generic_calls.get(), 1);
        assert_eq!(fold_calls.get(), 0);
    }
}
