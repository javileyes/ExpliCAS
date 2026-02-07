use crate::build::mul2_raw;
use crate::step::Step;
use cas_ast::{Context, ExprId};

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
        (Expr::Hold(inner), PathStep::Inner) => {
            let new_inner = reconstruct_global(ctx, inner, remaining_path, replacement);
            ctx.add(Expr::Hold(new_inner))
        }
        _ => root,
    }
}
