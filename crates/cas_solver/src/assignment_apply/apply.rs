use cas_ast::{Expr, ExprId};

use crate::assignment_apply::AssignmentApplyContext;
use crate::{AssignmentError, Simplifier};

fn unwrap_hold_top(ctx: &cas_ast::Context, expr: ExprId) -> ExprId {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if ctx.is_builtin(*name, cas_ast::BuiltinFn::Hold) && args.len() == 1 {
            return args[0];
        }
    }
    expr
}

/// Apply an assignment:
/// - `lazy = false` behaves like `let a = ...` (eager simplify + unwrap top-level hold).
/// - `lazy = true` behaves like `let a := ...` (store unresolved formula after ref/env substitution).
pub fn apply_assignment_with_context<C: AssignmentApplyContext>(
    context: &mut C,
    simplifier: &mut Simplifier,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<ExprId, AssignmentError> {
    if name.is_empty() {
        return Err(AssignmentError::EmptyName);
    }

    let starts_with_letter = name
        .chars()
        .next()
        .map(|c| c.is_alphabetic())
        .unwrap_or(false);
    if !starts_with_letter && !name.starts_with('_') {
        return Err(AssignmentError::InvalidNameStart);
    }

    if context.assignment_is_reserved_name(name) {
        return Err(AssignmentError::ReservedName(name.to_string()));
    }

    let rhs_expr = cas_parser::parse(expr_str, &mut simplifier.context)
        .map_err(|e| AssignmentError::Parse(e.to_string()))?;

    context.assignment_unset_binding(name);

    let rhs_substituted =
        match context.assignment_resolve_state_refs(&mut simplifier.context, rhs_expr) {
            Ok(resolved) => resolved,
            Err(_) => rhs_expr,
        };

    let result = if lazy {
        rhs_substituted
    } else {
        let (simplified, _steps) = simplifier.simplify(rhs_substituted);
        unwrap_hold_top(&simplifier.context, simplified)
    };

    context.assignment_set_binding(name.to_string(), result);
    Ok(result)
}
