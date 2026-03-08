use cas_ast::{Context, Equation, Expr, ExprId, RelOp};

/// Didactic payload for one linear-collect solve step.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearCollectDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// Pair of didactic steps emitted by linear-collect execution.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearCollectDidacticPair {
    pub collect: LinearCollectDidacticStep,
    pub divide: LinearCollectDidacticStep,
}

/// One executable linear-collect item aligned with a didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearCollectExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl LinearCollectExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

fn linear_collect_execution_item_from_didactic_step(
    didactic: LinearCollectDidacticStep,
) -> LinearCollectExecutionItem {
    LinearCollectExecutionItem {
        equation: didactic.equation_after,
        description: didactic.description,
    }
}

/// Convert linear-collect didactic pair into ordered step vector.
pub fn collect_linear_collect_didactic_steps(
    pair: LinearCollectDidacticPair,
) -> Vec<LinearCollectDidacticStep> {
    vec![pair.collect, pair.divide]
}

/// Convert linear-collect didactic pair into ordered execution items.
pub fn collect_linear_collect_execution_items(
    pair: LinearCollectDidacticPair,
) -> Vec<LinearCollectExecutionItem> {
    collect_linear_collect_didactic_steps(pair)
        .into_iter()
        .map(linear_collect_execution_item_from_didactic_step)
        .collect()
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

/// Build execution payload for the factored collect step (`coef*var = rhs`).
pub fn build_linear_collect_factored_execution_item_with<F>(
    ctx: &mut Context,
    var: &str,
    coef: ExprId,
    rhs: ExprId,
    render_expr: F,
) -> LinearCollectExecutionItem
where
    F: FnMut(&Context, ExprId) -> String,
{
    linear_collect_execution_item_from_didactic_step(build_linear_collect_factored_step_with(
        ctx,
        var,
        coef,
        rhs,
        render_expr,
    ))
}

/// Build execution payload for the additive collect step (`coef*var + c = 0`).
pub fn build_linear_collect_additive_execution_item(
    ctx: &mut Context,
    var: &str,
    coef: ExprId,
    constant: ExprId,
) -> LinearCollectExecutionItem {
    linear_collect_execution_item_from_didactic_step(build_linear_collect_additive_step(
        ctx, var, coef, constant,
    ))
}

/// Build execution payload for the divide step (`var = solution`).
pub fn build_linear_collect_divide_execution_item_with<F>(
    ctx: &mut Context,
    var: &str,
    solution: ExprId,
    coef: ExprId,
    mention_both_sides: bool,
    render_expr: F,
) -> LinearCollectExecutionItem
where
    F: FnMut(&Context, ExprId) -> String,
{
    linear_collect_execution_item_from_didactic_step(build_linear_collect_divide_step_with(
        ctx,
        var,
        solution,
        coef,
        mention_both_sides,
        render_expr,
    ))
}

/// Build factored + divide didactic steps for linear-collect flow:
/// `coef*var = rhs` then `var = solution`.
pub fn build_linear_collect_factored_steps_with<F>(
    ctx: &mut Context,
    var: &str,
    coef: ExprId,
    rhs: ExprId,
    solution: ExprId,
    mut render_expr: F,
) -> LinearCollectDidacticPair
where
    F: FnMut(&Context, ExprId) -> String,
{
    let collect = build_linear_collect_factored_step_with(ctx, var, coef, rhs, &mut render_expr);
    let divide = build_linear_collect_divide_step_with(ctx, var, solution, coef, true, render_expr);
    LinearCollectDidacticPair { collect, divide }
}

/// Build additive + divide didactic steps for linear-collect flow:
/// `coef*var + constant = 0` then `var = solution`.
pub fn build_linear_collect_additive_steps_with<F>(
    ctx: &mut Context,
    var: &str,
    coef: ExprId,
    constant: ExprId,
    solution: ExprId,
    render_expr: F,
) -> LinearCollectDidacticPair
where
    F: FnMut(&Context, ExprId) -> String,
{
    let collect = build_linear_collect_additive_step(ctx, var, coef, constant);
    let divide =
        build_linear_collect_divide_step_with(ctx, var, solution, coef, false, render_expr);
    LinearCollectDidacticPair { collect, divide }
}

/// Build factored + divide execution items for linear-collect flow:
/// `coef*var = rhs` then `var = solution`.
pub fn build_linear_collect_factored_execution_items_with<F>(
    ctx: &mut Context,
    var: &str,
    coef: ExprId,
    rhs: ExprId,
    solution: ExprId,
    mut render_expr: F,
) -> Vec<LinearCollectExecutionItem>
where
    F: FnMut(&Context, ExprId) -> String,
{
    let collect =
        build_linear_collect_factored_execution_item_with(ctx, var, coef, rhs, &mut render_expr);
    let divide = build_linear_collect_divide_execution_item_with(
        ctx,
        var,
        solution,
        coef,
        true,
        render_expr,
    );
    vec![collect, divide]
}

/// Build additive + divide execution items for linear-collect flow:
/// `coef*var + constant = 0` then `var = solution`.
pub fn build_linear_collect_additive_execution_items_with<F>(
    ctx: &mut Context,
    var: &str,
    coef: ExprId,
    constant: ExprId,
    solution: ExprId,
    render_expr: F,
) -> Vec<LinearCollectExecutionItem>
where
    F: FnMut(&Context, ExprId) -> String,
{
    let collect = build_linear_collect_additive_execution_item(ctx, var, coef, constant);
    let divide = build_linear_collect_divide_execution_item_with(
        ctx,
        var,
        solution,
        coef,
        false,
        render_expr,
    );
    vec![collect, divide]
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
            build_linear_collect_factored_step_with(&mut ctx, "x", coef, rhs, |_, _| "expr".into());
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
        let step =
            build_linear_collect_divide_step_with(&mut ctx, "x", solution, coef, true, |_, _| {
                "k".into()
            });
        assert_eq!(step.description, "Divide both sides by k");
        assert_eq!(step.equation_after.op, RelOp::Eq);
    }

    #[test]
    fn build_linear_collect_factored_steps_with_builds_pair() {
        let mut ctx = Context::new();
        let coef = ctx.var("k");
        let rhs = ctx.var("r");
        let solution = ctx.var("s");
        let pair =
            build_linear_collect_factored_steps_with(&mut ctx, "x", coef, rhs, solution, |_, _| {
                "expr".into()
            });
        assert_eq!(
            pair.collect.description,
            "Collect terms in x and factor: expr · x = expr"
        );
        assert_eq!(pair.divide.description, "Divide both sides by expr");
    }

    #[test]
    fn build_linear_collect_additive_steps_with_builds_pair() {
        let mut ctx = Context::new();
        let coef = ctx.var("k");
        let constant = ctx.var("c");
        let solution = ctx.var("s");
        let pair = build_linear_collect_additive_steps_with(
            &mut ctx,
            "x",
            coef,
            constant,
            solution,
            |_, _| "k".into(),
        );
        assert_eq!(pair.collect.description, "Collect terms in x");
        assert_eq!(pair.divide.description, "Divide by k");
    }

    #[test]
    fn collect_linear_collect_didactic_steps_preserves_collect_then_divide() {
        let mut ctx = Context::new();
        let coef = ctx.var("k");
        let rhs = ctx.var("r");
        let solution = ctx.var("s");
        let pair =
            build_linear_collect_factored_steps_with(&mut ctx, "x", coef, rhs, solution, |_, _| {
                "expr".into()
            });
        let steps = collect_linear_collect_didactic_steps(pair);

        assert_eq!(steps.len(), 2);
        assert_eq!(
            steps[0].description,
            "Collect terms in x and factor: expr · x = expr"
        );
        assert_eq!(steps[1].description, "Divide both sides by expr");
    }

    #[test]
    fn collect_linear_collect_execution_items_align_equations_with_didactic() {
        let mut ctx = Context::new();
        let coef = ctx.var("k");
        let rhs = ctx.var("r");
        let solution = ctx.var("s");
        let pair =
            build_linear_collect_factored_steps_with(&mut ctx, "x", coef, rhs, solution, |_, _| {
                "expr".into()
            });
        let didactic = collect_linear_collect_didactic_steps(pair.clone());

        let items = collect_linear_collect_execution_items(pair);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].equation, didactic[0].equation_after);
        assert_eq!(items[1].equation, didactic[1].equation_after);
        assert_eq!(items[0].description, didactic[0].description);
        assert_eq!(items[1].description, didactic[1].description);
    }

    #[test]
    fn build_linear_collect_factored_execution_items_with_builds_ordered_items() {
        let mut ctx = Context::new();
        let coef = ctx.var("k");
        let rhs = ctx.var("r");
        let solution = ctx.var("s");
        let items = build_linear_collect_factored_execution_items_with(
            &mut ctx,
            "x",
            coef,
            rhs,
            solution,
            |_, _| "expr".into(),
        );

        assert_eq!(items.len(), 2);
        assert_eq!(
            items[0].description,
            "Collect terms in x and factor: expr · x = expr"
        );
        assert_eq!(items[1].description, "Divide both sides by expr");
    }

    #[test]
    fn build_linear_collect_additive_execution_items_with_builds_ordered_items() {
        let mut ctx = Context::new();
        let coef = ctx.var("k");
        let constant = ctx.var("c");
        let solution = ctx.var("s");
        let items = build_linear_collect_additive_execution_items_with(
            &mut ctx,
            "x",
            coef,
            constant,
            solution,
            |_, _| "k".into(),
        );

        assert_eq!(items.len(), 2);
        assert_eq!(items[0].description, "Collect terms in x");
        assert_eq!(items[1].description, "Divide by k");
    }
}
