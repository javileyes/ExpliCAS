use crate::verification::VerifyStatus;
use crate::verify_substitution::substitute_equation_diff;
use cas_ast::{Context, Equation, ExprId};

/// Runtime contract for solution verification checks.
///
/// Implementors provide domain-mode simplification behavior and expression
/// predicates/rendering while solver-core owns the orchestration algorithm.
pub trait SolutionCheckRuntime {
    /// Mutable access to the expression context for substitution.
    fn context(&mut self) -> &mut Context;
    /// Simplify with strict, domain-honest semantics.
    fn simplify_strict(&mut self, expr: ExprId) -> ExprId;
    /// Simplify with generic fallback semantics.
    fn simplify_generic(&mut self, expr: ExprId) -> ExprId;
    /// Fold numeric islands inside one residual expression.
    fn fold_numeric_islands(&mut self, expr: ExprId) -> ExprId;
    /// Whether expression is numerically zero.
    fn is_numeric_zero(&mut self, expr: ExprId) -> bool;
    /// Whether expression still contains symbolic variables.
    fn contains_variable(&mut self, expr: ExprId) -> bool;
    /// Render expression for diagnostic messages.
    fn render_expr(&mut self, expr: ExprId) -> String;
}

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

/// Verify one candidate solution by substitution and two-phase simplification.
///
/// Algorithm:
/// 1. Substitute and form residual `lhs-rhs`.
/// 2. Strict simplify; accept if zero.
/// 3. If strict residual has variables, try numeric-island fold + strict recheck.
/// 4. If strict residual is variable-free, try generic fallback simplify.
/// 5. Return `Unverifiable` with strict residual when all checks fail.
pub fn verify_solution_with_runtime<R>(
    runtime: &mut R,
    equation: &Equation,
    var: &str,
    solution: ExprId,
) -> VerifyStatus
where
    R: SolutionCheckRuntime,
{
    let diff = {
        let ctx = runtime.context();
        substitute_equation_diff(ctx, equation, var, solution)
    };

    let strict_result = runtime.simplify_strict(diff);
    if runtime.is_numeric_zero(strict_result) {
        return VerifyStatus::Verified;
    }

    if runtime.contains_variable(strict_result) {
        crate::verify_stats::record_attempted();
        let folded = runtime.fold_numeric_islands(strict_result);
        if folded != strict_result {
            crate::verify_stats::record_changed();
            let folded_result = runtime.simplify_strict(folded);
            if runtime.is_numeric_zero(folded_result) {
                crate::verify_stats::record_verified();
                return VerifyStatus::Verified;
            }
        }
    }

    if !runtime.contains_variable(strict_result) {
        let generic_result = runtime.simplify_generic(diff);
        if runtime.is_numeric_zero(generic_result) {
            return VerifyStatus::Verified;
        }
    }

    VerifyStatus::Unverifiable {
        residual: strict_result,
        reason: format!("residual: {}", runtime.render_expr(strict_result)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Equation, RelOp};

    struct MockCheckRuntime {
        context: Context,
        strict_results: Vec<ExprId>,
        generic_result: ExprId,
        fold_result: ExprId,
        contains_var_for: Vec<ExprId>,
        zero_expr: ExprId,
        strict_calls: usize,
        generic_calls: usize,
        fold_calls: usize,
    }

    impl SolutionCheckRuntime for MockCheckRuntime {
        fn context(&mut self) -> &mut Context {
            &mut self.context
        }

        fn simplify_strict(&mut self, _expr: ExprId) -> ExprId {
            self.strict_calls += 1;
            self.strict_results.remove(0)
        }

        fn simplify_generic(&mut self, _expr: ExprId) -> ExprId {
            self.generic_calls += 1;
            self.generic_result
        }

        fn fold_numeric_islands(&mut self, _expr: ExprId) -> ExprId {
            self.fold_calls += 1;
            self.fold_result
        }

        fn is_numeric_zero(&mut self, expr: ExprId) -> bool {
            expr == self.zero_expr
        }

        fn contains_variable(&mut self, expr: ExprId) -> bool {
            self.contains_var_for.contains(&expr)
        }

        fn render_expr(&mut self, expr: ExprId) -> String {
            format!("expr#{:?}", expr)
        }
    }

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
    fn verify_solution_with_runtime_accepts_strict_zero() {
        let mut context = Context::new();
        let zero = context.num(0);
        let one = context.num(1);
        let (equation, solution) = sample_equation(&mut context);
        let mut runtime = MockCheckRuntime {
            context,
            strict_results: vec![zero],
            generic_result: one,
            fold_result: one,
            contains_var_for: vec![],
            zero_expr: zero,
            strict_calls: 0,
            generic_calls: 0,
            fold_calls: 0,
        };

        let out = verify_solution_with_runtime(&mut runtime, &equation, "x", solution);
        assert!(matches!(out, VerifyStatus::Verified));
        assert_eq!(runtime.strict_calls, 1);
        assert_eq!(runtime.generic_calls, 0);
        assert_eq!(runtime.fold_calls, 0);
    }

    #[test]
    fn verify_solution_with_runtime_accepts_after_phase_15_fold() {
        let mut context = Context::new();
        let x = context.var("x");
        let one = context.num(1);
        let zero = context.num(0);
        let folded = context.num(2);
        let (equation, solution) = sample_equation(&mut context);
        let mut runtime = MockCheckRuntime {
            context,
            strict_results: vec![x, zero],
            generic_result: one,
            fold_result: folded,
            contains_var_for: vec![x],
            zero_expr: zero,
            strict_calls: 0,
            generic_calls: 0,
            fold_calls: 0,
        };

        let out = verify_solution_with_runtime(&mut runtime, &equation, "x", solution);
        assert!(matches!(out, VerifyStatus::Verified));
        assert_eq!(runtime.strict_calls, 2);
        assert_eq!(runtime.generic_calls, 0);
        assert_eq!(runtime.fold_calls, 1);
    }

    #[test]
    fn verify_solution_with_runtime_accepts_after_generic_fallback() {
        let mut context = Context::new();
        let one = context.num(1);
        let two = context.num(2);
        let zero = context.num(0);
        let (equation, solution) = sample_equation(&mut context);
        let mut runtime = MockCheckRuntime {
            context,
            strict_results: vec![two],
            generic_result: zero,
            fold_result: one,
            contains_var_for: vec![],
            zero_expr: zero,
            strict_calls: 0,
            generic_calls: 0,
            fold_calls: 0,
        };

        let out = verify_solution_with_runtime(&mut runtime, &equation, "x", solution);
        assert!(matches!(out, VerifyStatus::Verified));
        assert_eq!(runtime.strict_calls, 1);
        assert_eq!(runtime.generic_calls, 1);
        assert_eq!(runtime.fold_calls, 0);
    }

    #[test]
    fn verify_solution_with_runtime_reports_unverifiable_when_checks_fail() {
        let mut context = Context::new();
        let one = context.num(1);
        let two = context.num(2);
        let three = context.num(3);
        let zero = context.num(0);
        let (equation, solution) = sample_equation(&mut context);
        let mut runtime = MockCheckRuntime {
            context,
            strict_results: vec![two],
            generic_result: three,
            fold_result: one,
            contains_var_for: vec![],
            zero_expr: zero,
            strict_calls: 0,
            generic_calls: 0,
            fold_calls: 0,
        };

        let out = verify_solution_with_runtime(&mut runtime, &equation, "x", solution);
        match out {
            VerifyStatus::Unverifiable { residual, reason } => {
                assert_eq!(residual, two);
                assert!(reason.contains("residual: expr#"));
            }
            other => panic!("expected unverifiable, got {:?}", other),
        }
        assert_eq!(runtime.strict_calls, 1);
        assert_eq!(runtime.generic_calls, 1);
        assert_eq!(runtime.fold_calls, 0);
    }
}
