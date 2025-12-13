use crate::build::mul2_raw;
use crate::collect::collect;
use crate::expand::expand;
use crate::factor::factor;
use crate::rule::Rule;
use crate::rules::algebra::SimplifyFractionRule;
use crate::rules::arithmetic::CombineConstantsRule;
use crate::rules::canonicalization::{CanonicalizeMulRule, CanonicalizeNegationRule};
use crate::rules::exponents::{EvaluatePowerRule, IdentityPowerRule, ProductPowerRule};
use crate::rules::polynomial::CombineLikeTermsRule;
use crate::step::{PathStep, Step};
use cas_ast::{Context, DisplayExpr, Expr, ExprId};

/// Helper: Build a 2-factor product (no normalization).
#[inline]

/// Strategy to simplify polynomials by trying expansion and factorization.
/// Returns the simplest form found.
pub fn simplify_polynomial(ctx: &mut Context, expr: ExprId) -> (ExprId, Vec<Step>) {
    let mut steps = Vec::new();

    // 1. Expand
    let expanded = expand(ctx, expr);
    if expanded != expr {
        steps.push(Step::new(
            "Expand Polynomial",
            "Expand",
            expr,
            expanded,
            Vec::new(),
            Some(ctx),
        ));
    }

    // 2. Clean up expansion (Collect + Combine Like Terms + Power Rules)
    let mut current = expanded;
    let mut i = 0;
    while i < 10 {
        let prev = current;

        // Collect first
        let collected = collect(ctx, current);
        if collected != current {
            // We could add a step here, but collect is often implicit.
            // Let's add it if it changes things significantly?
            // For consistency with orchestrator, let's skip explicit collect steps here
            // unless we want very detailed traces.
            // But since we are inside "Polynomial Strategy", maybe we want details?
            // Let's skip for now to avoid noise, as apply_rules_to_tree will show the main simplifications.
            current = collected;
        }

        // Apply rules recursively
        current = apply_rules_to_tree(ctx, current, &mut steps, Vec::new());

        if current == prev {
            break;
        }
        i += 1;
    }
    let simplified_expanded = current;

    // 3. Factor the result
    let factored = factor(ctx, simplified_expanded);

    // 4. Compare and choose best
    // Heuristic: Prefer 0, then prefer factored form if it has structure (Mul/Pow), then shortest.

    let s_orig = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: expr
        }
    );
    let s_exp = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: simplified_expanded
        }
    );
    let s_fact = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: factored
        }
    );

    let mut chosen = expr; // Default to original

    // 1. Prefer 0
    if s_exp == "0" {
        chosen = simplified_expanded;
    } else if s_fact == "0" {
        chosen = factored;
    } else {
        // 2. Prefer Factored if it's a Product or Power and Expanded is Sum
        let fact_data = ctx.get(factored);
        let exp_data = ctx.get(simplified_expanded);

        let is_fact_structured = matches!(
            fact_data,
            cas_ast::Expr::Mul(_, _) | cas_ast::Expr::Pow(_, _)
        );
        let is_exp_sum = matches!(
            exp_data,
            cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _)
        );

        if is_fact_structured && is_exp_sum {
            chosen = factored;
        } else {
            // 3. Length heuristic as fallback
            let len_orig = s_orig.len();
            let len_exp = s_exp.len();
            let len_fact = s_fact.len();

            // Prefer factored if significantly shorter or same length
            if len_fact <= len_exp && len_fact <= len_orig {
                chosen = factored;
            } else if len_exp < len_orig {
                chosen = simplified_expanded;
            }
        }
    }

    if chosen == factored {
        if factored != simplified_expanded {
            steps.push(Step::new(
                "Factor Polynomial",
                "Factor",
                simplified_expanded,
                factored,
                Vec::new(),
                Some(ctx),
            ));
        }
        (factored, filter_non_productive_steps(ctx, expr, steps))
    } else if chosen == simplified_expanded {
        (
            simplified_expanded,
            filter_non_productive_steps(ctx, expr, steps),
        )
    } else {
        // Reverted to original
        (expr, Vec::new())
    }
}

/// Filter out steps that don't change the global expression state
pub fn filter_non_productive_steps(
    ctx: &mut Context,
    original: ExprId,
    steps: Vec<Step>,
) -> Vec<Step> {
    let mut filtered = Vec::new();
    let mut current_global = original;

    for step in steps {
        // FIRST: Check for didactically important steps that should always be kept
        // These bypass ALL filtering because they're pedagogically valuable
        if step.rule_name == "Sum Exponents" || step.rule_name == "Evaluate Numeric Power" {
            let global_after = reconstruct_global(ctx, current_global, &step.path, step.after);
            filtered.push(step);
            current_global = global_after;
            continue;
        }

        // Filter local no-op steps: if step.before == step.after semantically, skip
        {
            let checker = crate::semantic_equality::SemanticEqualityChecker::new(ctx);
            if checker.are_equal(step.before, step.after) {
                // Local no-op - don't include this step
                continue;
            }
        }

        // Fallback: compare display strings - if they look identical to user, it's a no-op
        {
            let before_str = format!(
                "{}",
                cas_ast::DisplayExpr {
                    context: ctx,
                    id: step.before
                }
            );
            let after_str = format!(
                "{}",
                cas_ast::DisplayExpr {
                    context: ctx,
                    id: step.after
                }
            );
            if before_str == after_str {
                // Display no-op - same string representation
                continue;
            }
        }

        let global_after = reconstruct_global(ctx, current_global, &step.path, step.after);
        let checker = crate::semantic_equality::SemanticEqualityChecker::new(ctx);
        if !checker.are_equal(current_global, global_after) {
            filtered.push(step);
            current_global = global_after;
        }
    }

    filtered
}

/// Reconstruct global expression by applying local change at path
fn reconstruct_global(
    ctx: &mut Context,
    root: ExprId,
    path: &[crate::step::PathStep],
    replacement: ExprId,
) -> ExprId {
    if path.is_empty() {
        return replacement;
    }

    use crate::step::PathStep;
    use cas_ast::Expr;

    let current_step = &path[0];
    let remaining_path = &path[1..];
    let expr = ctx.get(root).clone();

    match (expr, current_step) {
        (Expr::Add(l, r), PathStep::Left) => {
            let new_l = reconstruct_global(ctx, l, remaining_path, replacement);
            ctx.add(Expr::Add(new_l, r))
        }
        (Expr::Add(l, r), PathStep::Right) => {
            let new_r = reconstruct_global(ctx, r, remaining_path, replacement);
            ctx.add(Expr::Add(l, new_r))
        }
        (Expr::Sub(l, r), PathStep::Left) => {
            let new_l = reconstruct_global(ctx, l, remaining_path, replacement);
            ctx.add(Expr::Sub(new_l, r))
        }
        (Expr::Sub(l, r), PathStep::Right) => {
            let new_r = reconstruct_global(ctx, r, remaining_path, replacement);
            ctx.add(Expr::Sub(l, new_r))
        }
        (Expr::Mul(l, r), PathStep::Left) => {
            let new_l = reconstruct_global(ctx, l, remaining_path, replacement);
            mul2_raw(ctx, new_l, r)
        }
        (Expr::Mul(l, r), PathStep::Right) => {
            let new_r = reconstruct_global(ctx, r, remaining_path, replacement);
            mul2_raw(ctx, l, new_r)
        }
        (Expr::Div(l, r), PathStep::Left) => {
            let new_l = reconstruct_global(ctx, l, remaining_path, replacement);
            ctx.add(Expr::Div(new_l, r))
        }
        (Expr::Div(l, r), PathStep::Right) => {
            let new_r = reconstruct_global(ctx, r, remaining_path, replacement);
            ctx.add(Expr::Div(l, new_r))
        }
        (Expr::Pow(b, e), PathStep::Base) => {
            let new_b = reconstruct_global(ctx, b, remaining_path, replacement);
            ctx.add(Expr::Pow(new_b, e))
        }
        (Expr::Pow(b, e), PathStep::Exponent) => {
            let new_e = reconstruct_global(ctx, e, remaining_path, replacement);
            ctx.add(Expr::Pow(b, new_e))
        }
        (Expr::Neg(e), PathStep::Inner) => {
            let new_e = reconstruct_global(ctx, e, remaining_path, replacement);
            ctx.add(Expr::Neg(new_e))
        }
        (Expr::Function(name, args), PathStep::Arg(idx)) => {
            let mut new_args = args.clone();
            if *idx < new_args.len() {
                new_args[*idx] = reconstruct_global(ctx, args[*idx], remaining_path, replacement);
                ctx.add(Expr::Function(name, new_args))
            } else {
                root
            }
        }
        _ => root,
    }
}

fn apply_rules_to_tree(
    ctx: &mut Context,
    expr: ExprId,
    steps: &mut Vec<Step>,
    path: Vec<PathStep>,
) -> ExprId {
    use cas_ast::Expr;

    // 1. Recurse on children
    let expr_data = ctx.get(expr).clone();
    let mut new_expr = match expr_data {
        Expr::Add(l, r) => {
            let mut p_l = path.clone();
            p_l.push(PathStep::Left);
            let nl = apply_rules_to_tree(ctx, l, steps, p_l);

            let mut p_r = path.clone();
            p_r.push(PathStep::Right);
            let nr = apply_rules_to_tree(ctx, r, steps, p_r);

            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Expr::Sub(l, r) => {
            let mut p_l = path.clone();
            p_l.push(PathStep::Left);
            let nl = apply_rules_to_tree(ctx, l, steps, p_l);

            let mut p_r = path.clone();
            p_r.push(PathStep::Right);
            let nr = apply_rules_to_tree(ctx, r, steps, p_r);

            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }
        Expr::Mul(l, r) => {
            let mut p_l = path.clone();
            p_l.push(PathStep::Left);
            let nl = apply_rules_to_tree(ctx, l, steps, p_l);

            let mut p_r = path.clone();
            p_r.push(PathStep::Right);
            let nr = apply_rules_to_tree(ctx, r, steps, p_r);

            if nl != l || nr != r {
                mul2_raw(ctx, nl, nr)
            } else {
                expr
            }
        }
        Expr::Div(l, r) => {
            let mut p_l = path.clone();
            p_l.push(PathStep::Left);
            let nl = apply_rules_to_tree(ctx, l, steps, p_l);

            let mut p_r = path.clone();
            p_r.push(PathStep::Right);
            let nr = apply_rules_to_tree(ctx, r, steps, p_r);

            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Expr::Pow(b, e) => {
            let mut p_b = path.clone();
            p_b.push(PathStep::Base);
            let nb = apply_rules_to_tree(ctx, b, steps, p_b);

            let mut p_e = path.clone();
            p_e.push(PathStep::Exponent);
            let ne = apply_rules_to_tree(ctx, e, steps, p_e);

            if nb != b || ne != e {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                expr
            }
        }
        Expr::Neg(e) => {
            let mut p_e = path.clone();
            p_e.push(PathStep::Inner);
            let ne = apply_rules_to_tree(ctx, e, steps, p_e);

            if ne != e {
                ctx.add(Expr::Neg(ne))
            } else {
                expr
            }
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::new();
            let mut changed = false;
            for (i, arg) in args.iter().enumerate() {
                let mut p_arg = path.clone();
                p_arg.push(PathStep::Arg(i));
                let new_arg = apply_rules_to_tree(ctx, *arg, steps, p_arg);
                if new_arg != *arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }
        _ => expr,
    };

    // 2. Apply rules to self
    let mut changed = true;
    while changed {
        changed = false;

        let mut apply_rule = |rule: &dyn Rule, current_expr: ExprId| -> Option<ExprId> {
            if let Some(rw) =
                crate::semantic_equality::apply_rule_with_semantic_check(ctx, rule, current_expr)
            {
                let mut step = Step::new(
                    &rw.description,
                    rule.name(),
                    current_expr,
                    rw.new_expr,
                    path.clone(),
                    Some(ctx),
                );
                // Propagate local before/after from Rewrite for accurate Rule display
                step.before_local = rw.before_local;
                step.after_local = rw.after_local;
                steps.push(step);
                Some(rw.new_expr)
            } else {
                None
            }
        };

        // Canonicalize Negation first (important for combining)
        if let Some(res) = apply_rule(&CanonicalizeNegationRule, new_expr) {
            new_expr = res;
            changed = true;
        }

        // Canonicalize Multiplication (sorts factors for ProductPowerRule)
        if let Some(res) = apply_rule(&CanonicalizeMulRule, new_expr) {
            new_expr = res;
            changed = true;
        }

        // Power Rules
        if let Some(res) = apply_rule(&ProductPowerRule, new_expr) {
            new_expr = res;
            changed = true;
        }
        if let Some(res) = apply_rule(&IdentityPowerRule, new_expr) {
            new_expr = res;
            changed = true;
        }
        if let Some(res) = apply_rule(&EvaluatePowerRule, new_expr) {
            new_expr = res;
            changed = true;
        }

        if let Some(res) = apply_rule(&CombineLikeTermsRule, new_expr) {
            new_expr = res;
            changed = true;
        }

        if let Some(res) = apply_rule(&CombineConstantsRule, new_expr) {
            new_expr = res;
            changed = true;
        }

        if let Some(res) = apply_rule(&SimplifyFractionRule, new_expr) {
            new_expr = res;
            changed = true;
        }
    }
    new_expr
}
