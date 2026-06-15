//! Local active solve backend boundary.
//!
//! This backend is solver-owned and executes the solver-native runtime pipeline.
//! Keeping this indirection local lets us switch implementations without
//! changing call sites.

use crate::{CasError, Simplifier, SolveCtx, SolveStep};
use cas_ast::{Constant, Context, Equation, Expr, ExprId, SolutionSet};

use crate::solve_backend_contract::{CoreSolverOptions, SolveBackend};

/// True when `expr` contains a non-finite / undefined constant (∞ or undefined)
/// anywhere. Such a value is never a real solution of an equation over ℝ.
fn solution_contains_nonfinite(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity | Constant::Undefined) => true,
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            solution_contains_nonfinite(ctx, *a) || solution_contains_nonfinite(ctx, *b)
        }
        Expr::Neg(a) | Expr::Hold(a) => solution_contains_nonfinite(ctx, *a),
        Expr::Function(_, args) => args.iter().any(|&c| solution_contains_nonfinite(ctx, c)),
        Expr::Matrix { data, .. } => data.iter().any(|&c| solution_contains_nonfinite(ctx, c)),
        _ => false,
    }
}

/// Drop non-finite (∞ / undefined) entries from the final real solution set:
/// `solve(3/x = 0)` has NO real solution, not `{∞}` (the solver reaches `∞` by
/// dividing by zero while isolating the variable). An emptied discrete set
/// collapses to `Empty` (no solution). Recurses into conditional cases.
fn drop_nonfinite_solutions(ctx: &Context, set: SolutionSet) -> SolutionSet {
    match set {
        SolutionSet::Discrete(sols) => {
            let kept: Vec<ExprId> = sols
                .into_iter()
                .filter(|&s| !solution_contains_nonfinite(ctx, s))
                .collect();
            if kept.is_empty() {
                SolutionSet::Empty
            } else {
                SolutionSet::Discrete(kept)
            }
        }
        SolutionSet::Conditional(cases) => SolutionSet::Conditional(
            cases
                .into_iter()
                .map(|mut case| {
                    let filtered = drop_nonfinite_solutions(ctx, case.then.solutions.clone());
                    case.then.solutions = filtered;
                    case
                })
                .collect(),
        ),
        other => other,
    }
}

/// True when the equation is `c/poly = 0` for a nonzero constant `c` — which has
/// NO real solution (a nonzero constant over anything is never zero; the points
/// where the denominator vanishes make it undefined, not zero). Detected by
/// simplifying `lhs - rhs` to a single fraction with a nonzero-constant numerator.
///
/// Short-circuiting this BEFORE the isolation logic avoids the solver dividing by
/// zero (`poly = c/0 = ∞`), which otherwise fabricates `{∞}` or a malformed nested
/// `solve(x = ∞ - x^2, x)` for denominators like `x^2 + x + 1`.
fn equation_is_nonzero_const_over_polynomial(simplifier: &mut Simplifier, eq: &Equation) -> bool {
    use num_traits::Zero;
    if eq.op != cas_ast::RelOp::Eq {
        return false;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (simplified, _) = simplifier.simplify(diff);
    let Expr::Div(num, _den) = simplifier.context.get(simplified) else {
        return false;
    };
    let num = *num;
    cas_math::numeric_eval::as_rational_const(&simplifier.context, num)
        .is_some_and(|r| !r.is_zero())
}

/// Local backend facade selected as the active backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalSolveBackend;

impl SolveBackend for LocalSolveBackend {
    fn solve_with_ctx_and_options(
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        opts: CoreSolverOptions,
        ctx: &SolveCtx,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        if equation_is_nonzero_const_over_polynomial(simplifier, eq) {
            return Ok((SolutionSet::Empty, Vec::new()));
        }
        let (set, steps) = crate::solve_core_runtime::solve_inner(eq, var, simplifier, opts, ctx)?;
        let set = drop_nonfinite_solutions(&simplifier.context, set);
        Ok((set, steps))
    }
}
