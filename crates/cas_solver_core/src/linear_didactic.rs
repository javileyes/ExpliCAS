use cas_ast::{Context, Equation, Expr, ExprId, RelOp};

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
}
