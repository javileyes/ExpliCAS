use cas_ast::{Context, Equation, Expr, ExprId, RelOp};

/// Didactic payload for one linear-collect solve step.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearCollectDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// Build narration for the factored linear-collect step:
/// `coef * var = rhs`.
pub fn linear_collect_factored_message(
    var: &str,
    coeff_display: &str,
    rhs_display: &str,
) -> String {
    format!(
        "Collect terms in {} and factor: {} · {} = {}",
        var, coeff_display, var, rhs_display
    )
}

/// Build narration for linear-collect divide steps.
pub fn linear_collect_divide_message(coeff_display: &str, mention_both_sides: bool) -> String {
    if mention_both_sides {
        format!("Divide both sides by {}", coeff_display)
    } else {
        format!("Divide by {}", coeff_display)
    }
}

/// Build narration for the additive collect step before division.
pub fn linear_collect_collect_message(var: &str) -> String {
    format!("Collect terms in {}", var)
}

/// Build equation `coef * var = rhs`.
pub fn build_linear_collect_factored_equation(
    ctx: &mut Context,
    var: &str,
    coef: ExprId,
    rhs: ExprId,
) -> Equation {
    let var_id = ctx.var(var);
    Equation {
        lhs: ctx.add(Expr::Mul(coef, var_id)),
        rhs,
        op: RelOp::Eq,
    }
}

/// Build equation `coef * var + constant = 0`.
pub fn build_linear_collect_additive_equation(
    ctx: &mut Context,
    var: &str,
    coef: ExprId,
    constant: ExprId,
) -> Equation {
    let var_id = ctx.var(var);
    let coef_times_var = ctx.add(Expr::Mul(coef, var_id));
    Equation {
        lhs: ctx.add(Expr::Add(coef_times_var, constant)),
        rhs: ctx.num(0),
        op: RelOp::Eq,
    }
}

/// Build equation `var = solution`.
pub fn build_linear_collect_solution_equation(
    ctx: &mut Context,
    var: &str,
    solution: ExprId,
) -> Equation {
    Equation {
        lhs: ctx.var(var),
        rhs: solution,
        op: RelOp::Eq,
    }
}

/// Build didactic payload for the factored collect step (`coef*var = rhs`).
pub fn build_linear_collect_factored_step_with<F>(
    ctx: &mut Context,
    var: &str,
    coef: ExprId,
    rhs: ExprId,
    mut render_expr: F,
) -> LinearCollectDidacticStep
where
    F: FnMut(&Context, ExprId) -> String,
{
    let coeff_desc = render_expr(ctx, coef);
    let rhs_desc = render_expr(ctx, rhs);
    LinearCollectDidacticStep {
        description: linear_collect_factored_message(var, &coeff_desc, &rhs_desc),
        equation_after: build_linear_collect_factored_equation(ctx, var, coef, rhs),
    }
}

/// Build didactic payload for the additive collect step (`coef*var + c = 0`).
pub fn build_linear_collect_additive_step(
    ctx: &mut Context,
    var: &str,
    coef: ExprId,
    constant: ExprId,
) -> LinearCollectDidacticStep {
    LinearCollectDidacticStep {
        description: linear_collect_collect_message(var),
        equation_after: build_linear_collect_additive_equation(ctx, var, coef, constant),
    }
}

/// Build didactic payload for the divide step (`var = solution`).
pub fn build_linear_collect_divide_step_with<F>(
    ctx: &mut Context,
    var: &str,
    solution: ExprId,
    coef: ExprId,
    mention_both_sides: bool,
    mut render_expr: F,
) -> LinearCollectDidacticStep
where
    F: FnMut(&Context, ExprId) -> String,
{
    let coeff_desc = render_expr(ctx, coef);
    LinearCollectDidacticStep {
        description: linear_collect_divide_message(&coeff_desc, mention_both_sides),
        equation_after: build_linear_collect_solution_equation(ctx, var, solution),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_collect_factored_message_formats_expected_text() {
        let msg = linear_collect_factored_message("x", "a+b", "c");
        assert_eq!(msg, "Collect terms in x and factor: a+b · x = c");
    }

    #[test]
    fn linear_collect_divide_message_supports_both_styles() {
        assert_eq!(
            linear_collect_divide_message("k", true),
            "Divide both sides by k"
        );
        assert_eq!(linear_collect_divide_message("k", false), "Divide by k");
    }

    #[test]
    fn linear_collect_collect_message_formats_expected_text() {
        assert_eq!(linear_collect_collect_message("x"), "Collect terms in x");
    }

    #[test]
    fn build_linear_collect_solution_equation_uses_eq_relation() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let eq = build_linear_collect_solution_equation(&mut ctx, "x", y);
        assert_eq!(eq.op, RelOp::Eq);
        assert_eq!(eq.rhs, y);
    }

    #[test]
    fn build_linear_collect_factored_step_with_builds_payload() {
        let mut ctx = Context::new();
        let coef = ctx.var("k");
        let rhs = ctx.var("r");
        let step =
            build_linear_collect_factored_step_with(&mut ctx, "x", coef, rhs, |_, _| {
                "expr".into()
            });
        assert_eq!(
            step.description,
            "Collect terms in x and factor: expr · x = expr"
        );
        assert_eq!(step.equation_after.op, RelOp::Eq);
    }

    #[test]
    fn build_linear_collect_additive_step_builds_payload() {
        let mut ctx = Context::new();
        let coef = ctx.var("k");
        let constant = ctx.var("c");
        let step = build_linear_collect_additive_step(&mut ctx, "x", coef, constant);
        assert_eq!(step.description, "Collect terms in x");
        assert_eq!(step.equation_after.op, RelOp::Eq);
    }

    #[test]
    fn build_linear_collect_divide_step_with_builds_payload() {
        let mut ctx = Context::new();
        let coef = ctx.var("k");
        let solution = ctx.var("s");
        let step = build_linear_collect_divide_step_with(
            &mut ctx,
            "x",
            solution,
            coef,
            true,
            |_, _| "k".into(),
        );
        assert_eq!(step.description, "Divide both sides by k");
        assert_eq!(step.equation_after.op, RelOp::Eq);
    }
}
