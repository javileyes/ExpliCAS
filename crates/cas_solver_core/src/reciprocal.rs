use crate::isolation_utils::{contains_var, is_simple_reciprocal};
use crate::linear_solution::NonZeroStatus;
use cas_ast::{
    Case, ConditionPredicate, ConditionSet, Context, Equation, Expr, ExprId, RelOp, SolutionSet,
};
use cas_math::expr_nary::add_terms_no_sign;
use cas_math::expr_predicates::is_one_expr as is_one;

/// Represents a fraction as (numerator, denominator).
#[derive(Debug, Clone)]
struct Fraction {
    num: ExprId,
    den: ExprId,
}

/// Convert an expression to fraction form.
/// - `a/b` -> (a, b)
/// - `a` -> (a, 1)
fn expr_to_fraction(ctx: &mut Context, expr: ExprId) -> Fraction {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => Fraction { num, den },
        _ => {
            let one = ctx.num(1);
            Fraction {
                num: expr,
                den: one,
            }
        }
    }
}

/// Build scale factor for a fraction: product of all OTHER denominators.
fn build_scale_factor(ctx: &mut Context, fractions: &[Fraction], my_den: ExprId) -> ExprId {
    let other_dens: Vec<ExprId> = fractions
        .iter()
        .filter(|f| f.den != my_den)
        .map(|f| f.den)
        .collect();

    if other_dens.is_empty() {
        ctx.num(1)
    } else if other_dens.len() == 1 {
        other_dens[0]
    } else {
        let mut product = other_dens[0];
        for &d in &other_dens[1..] {
            product = ctx.add(Expr::Mul(product, d));
        }
        product
    }
}

/// Combine multiple fractions into a single fraction (numerator, denominator).
///
/// Uses common denominator `D = ∏ den_i`.
/// Numerator `N = Σ (num_i × (D/den_i))`.
///
/// Returns `(N, D)` after light structural normalization.
pub fn combine_fractions_deterministic(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let terms = add_terms_no_sign(ctx, expr);
    if terms.is_empty() {
        return None;
    }

    let fractions: Vec<Fraction> = terms.iter().map(|&t| expr_to_fraction(ctx, t)).collect();

    let common_denom = if fractions.len() == 1 {
        fractions[0].den
    } else {
        let mut denom = fractions[0].den;
        for frac in &fractions[1..] {
            denom = ctx.add(Expr::Mul(denom, frac.den));
        }
        denom
    };

    let mut scaled_nums: Vec<ExprId> = Vec::new();
    for frac in &fractions {
        let scale_factor = build_scale_factor(ctx, &fractions, frac.den);
        let scaled_num = if is_one(ctx, scale_factor) {
            frac.num
        } else {
            ctx.add(Expr::Mul(frac.num, scale_factor))
        };
        scaled_nums.push(scaled_num);
    }

    let numerator = if scaled_nums.len() == 1 {
        scaled_nums[0]
    } else {
        let mut sum = scaled_nums[0];
        for &term in &scaled_nums[1..] {
            sum = ctx.add(Expr::Add(sum, term));
        }
        sum
    };

    Some((numerator, common_denom))
}

/// Normalized data for reciprocal solve `1/var = rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReciprocalSolveKernel {
    pub numerator: ExprId,
    pub denominator: ExprId,
}

/// Human-readable step description for combining RHS fractions.
pub const RECIPROCAL_COMBINE_STEP_DESCRIPTION: &str =
    "Combine fractions on RHS (common denominator)";

/// Human-readable step description for reciprocal inversion.
pub const RECIPROCAL_INVERT_STEP_DESCRIPTION: &str = "Take reciprocal";

/// Didactic payload for one reciprocal-solve step.
#[derive(Debug, Clone, PartialEq)]
pub struct ReciprocalDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// Planned equation targets for reciprocal solve `1/var = rhs`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReciprocalSolvePlan {
    pub combine_equation: Equation,
    pub solve_equation: Equation,
    pub combined_rhs: ExprId,
    pub solution_rhs: ExprId,
}

/// Engine-facing reciprocal solve execution payload.
#[derive(Debug, Clone, PartialEq)]
pub struct ReciprocalSolveExecution {
    pub combine_step: ReciprocalDidacticStep,
    pub invert_step: ReciprocalDidacticStep,
    pub solutions: SolutionSet,
}

/// Derive reciprocal-solve kernel if equation matches `1/var = rhs` and
/// the RHS is independent of `var`.
pub fn derive_reciprocal_solve_kernel(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<ReciprocalSolveKernel> {
    if !is_simple_reciprocal(ctx, lhs, var) {
        return None;
    }
    if contains_var(ctx, rhs, var) {
        return None;
    }
    let (numerator, denominator) = combine_fractions_deterministic(ctx, rhs)?;
    Some(ReciprocalSolveKernel {
        numerator,
        denominator,
    })
}

/// Build the didactic execution plan for reciprocal solve.
///
/// Given normalized `rhs = numerator / denominator`, this returns:
/// 1. `1/var = numerator/denominator`
/// 2. `var = denominator/numerator`
pub fn build_reciprocal_solve_plan(
    ctx: &mut Context,
    var: &str,
    numerator: ExprId,
    denominator: ExprId,
) -> ReciprocalSolvePlan {
    let var_id = ctx.var(var);
    let one = ctx.num(1);
    let reciprocal_lhs = ctx.add(Expr::Div(one, var_id));
    let combined_rhs = ctx.add(Expr::Div(numerator, denominator));
    let solution_rhs = ctx.add(Expr::Div(denominator, numerator));

    let combine_equation = Equation {
        lhs: reciprocal_lhs,
        rhs: combined_rhs,
        op: RelOp::Eq,
    };
    let solve_equation = Equation {
        lhs: var_id,
        rhs: solution_rhs,
        op: RelOp::Eq,
    };

    ReciprocalSolvePlan {
        combine_equation,
        solve_equation,
        combined_rhs,
        solution_rhs,
    }
}

/// Build didactic payload for the reciprocal combine step.
pub fn build_reciprocal_combine_step(equation_after: Equation) -> ReciprocalDidacticStep {
    ReciprocalDidacticStep {
        description: RECIPROCAL_COMBINE_STEP_DESCRIPTION.to_string(),
        equation_after,
    }
}

/// Build didactic payload for the reciprocal inversion step.
pub fn build_reciprocal_invert_step(equation_after: Equation) -> ReciprocalDidacticStep {
    ReciprocalDidacticStep {
        description: RECIPROCAL_INVERT_STEP_DESCRIPTION.to_string(),
        equation_after,
    }
}

/// Build solution set for reciprocal equations `1/x = N/D` where
/// candidate solution is `x = D/N` and the domain requires `N != 0`.
pub fn build_reciprocal_solution_set(
    numerator: ExprId,
    solution: ExprId,
    numerator_status: NonZeroStatus,
) -> SolutionSet {
    if numerator_status == NonZeroStatus::NonZero {
        return SolutionSet::Discrete(vec![solution]);
    }

    let guard = ConditionSet::single(ConditionPredicate::NonZero(numerator));
    let case = Case::new(guard, SolutionSet::Discrete(vec![solution]));
    SolutionSet::Conditional(vec![case])
}

/// Build full reciprocal solve execution payload from prepared display/proof data.
pub fn build_reciprocal_execution(
    ctx: &mut Context,
    var: &str,
    numerator: ExprId,
    denominator: ExprId,
    combined_rhs_display: ExprId,
    solution_rhs_display: ExprId,
    guard_numerator: ExprId,
    numerator_status: NonZeroStatus,
) -> ReciprocalSolveExecution {
    let plan = build_reciprocal_solve_plan(ctx, var, numerator, denominator);

    let mut combine_equation = plan.combine_equation;
    combine_equation.rhs = combined_rhs_display;
    let combine_step = build_reciprocal_combine_step(combine_equation);

    let mut solve_equation = plan.solve_equation;
    solve_equation.rhs = solution_rhs_display;
    let invert_step = build_reciprocal_invert_step(solve_equation);

    let solutions =
        build_reciprocal_solution_set(guard_numerator, solution_rhs_display, numerator_status);

    ReciprocalSolveExecution {
        combine_step,
        invert_step,
        solutions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isolation_utils::{contains_var, is_simple_reciprocal};

    #[test]
    fn test_combine_fractions_simple() {
        let mut ctx = Context::new();
        let r1 = ctx.var("R1");
        let r2 = ctx.var("R2");
        let one = ctx.num(1);

        let frac1 = ctx.add(Expr::Div(one, r1));
        let one2 = ctx.num(1);
        let frac2 = ctx.add(Expr::Div(one2, r2));
        let sum = ctx.add(Expr::Add(frac1, frac2));

        let result = combine_fractions_deterministic(&mut ctx, sum);
        assert!(result.is_some());

        let (num, denom) = result.expect("must combine into a single fraction");
        assert!(contains_var(&ctx, num, "R1") || contains_var(&ctx, num, "R2"));
        assert!(contains_var(&ctx, denom, "R1"));
        assert!(contains_var(&ctx, denom, "R2"));
    }

    #[test]
    fn reciprocal_solution_nonzero_is_discrete() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let sol = ctx.var("x0");
        let set = build_reciprocal_solution_set(num, sol, NonZeroStatus::NonZero);
        assert_eq!(set, SolutionSet::Discrete(vec![sol]));
    }

    #[test]
    fn reciprocal_solution_unknown_is_conditional() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let sol = ctx.var("x0");
        let set = build_reciprocal_solution_set(num, sol, NonZeroStatus::Unknown);
        assert!(matches!(set, SolutionSet::Conditional(_)));
    }

    #[test]
    fn derive_reciprocal_solve_kernel_matches_simple_case() {
        let mut ctx = Context::new();
        let r = ctx.var("R");
        let r1 = ctx.var("R1");
        let r2 = ctx.var("R2");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Div(one, r));
        let one2 = ctx.num(1);
        let frac1 = ctx.add(Expr::Div(one2, r1));
        let one3 = ctx.num(1);
        let frac2 = ctx.add(Expr::Div(one3, r2));
        let rhs = ctx.add(Expr::Add(frac1, frac2));

        let kernel = derive_reciprocal_solve_kernel(&mut ctx, lhs, rhs, "R")
            .expect("must derive reciprocal kernel");
        assert!(
            contains_var(&ctx, kernel.numerator, "R1")
                || contains_var(&ctx, kernel.numerator, "R2")
        );
        assert!(contains_var(&ctx, kernel.denominator, "R1"));
        assert!(contains_var(&ctx, kernel.denominator, "R2"));
    }

    #[test]
    fn derive_reciprocal_solve_kernel_rejects_rhs_with_var() {
        let mut ctx = Context::new();
        let r = ctx.var("R");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Div(one, r));
        let rhs = r;
        assert!(is_simple_reciprocal(&ctx, lhs, "R"));
        assert!(derive_reciprocal_solve_kernel(&mut ctx, lhs, rhs, "R").is_none());
    }

    #[test]
    fn build_reciprocal_solve_plan_constructs_expected_shapes() {
        let mut ctx = Context::new();
        let numerator = ctx.var("n");
        let denominator = ctx.var("d");
        let plan = build_reciprocal_solve_plan(&mut ctx, "x", numerator, denominator);

        assert_eq!(plan.combine_equation.op, RelOp::Eq);
        assert_eq!(plan.solve_equation.op, RelOp::Eq);
        assert!(matches!(
            ctx.get(plan.combine_equation.lhs),
            Expr::Div(_, _)
        ));
        assert!(matches!(
            ctx.get(plan.combine_equation.rhs),
            Expr::Div(_, _)
        ));
        assert!(matches!(ctx.get(plan.solve_equation.rhs), Expr::Div(_, _)));
    }

    #[test]
    fn reciprocal_step_builders_use_standard_messages() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };
        let combine = build_reciprocal_combine_step(eq.clone());
        assert_eq!(
            combine.description,
            "Combine fractions on RHS (common denominator)"
        );
        assert_eq!(combine.equation_after, eq);

        let invert = build_reciprocal_invert_step(eq.clone());
        assert_eq!(invert.description, "Take reciprocal");
        assert_eq!(invert.equation_after, eq);
    }

    #[test]
    fn build_reciprocal_execution_assembles_steps_and_solution() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let combined_rhs = ctx.var("c");
        let solution_rhs = ctx.var("s");

        let execution = build_reciprocal_execution(
            &mut ctx,
            "x",
            num,
            den,
            combined_rhs,
            solution_rhs,
            num,
            NonZeroStatus::Unknown,
        );

        assert_eq!(
            execution.combine_step.description,
            "Combine fractions on RHS (common denominator)"
        );
        assert_eq!(execution.combine_step.equation_after.rhs, combined_rhs);
        assert_eq!(execution.invert_step.description, "Take reciprocal");
        assert_eq!(execution.invert_step.equation_after.rhs, solution_rhs);
        assert!(matches!(execution.solutions, SolutionSet::Conditional(_)));
    }
}
