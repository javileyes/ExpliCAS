use cas_ast::{Expr, ExprId};

use crate::assignment_apply::AssignmentApplyContext;
use crate::{AssignmentError, Simplifier};

#[derive(Debug, Clone, PartialEq, Eq)]
enum AssignmentTarget {
    Variable { name: String },
    Function { name: String, params: Vec<String> },
}

fn unwrap_hold_top(ctx: &cas_ast::Context, expr: ExprId) -> ExprId {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if ctx.is_builtin(*name, cas_ast::BuiltinFn::Hold) && args.len() == 1 {
            return args[0];
        }
    }
    expr
}

fn validate_identifier<C: AssignmentApplyContext>(
    context: &C,
    name: &str,
    reserved_error_name: &str,
) -> Result<(), AssignmentError> {
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

    if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return Err(AssignmentError::Parse(format!(
            "invalid assignment target '{}'",
            reserved_error_name
        )));
    }

    if context.assignment_is_reserved_name(name) {
        return Err(AssignmentError::ReservedName(
            reserved_error_name.to_string(),
        ));
    }

    Ok(())
}

fn parse_assignment_target<C: AssignmentApplyContext>(
    context: &C,
    raw: &str,
) -> Result<AssignmentTarget, AssignmentError> {
    let target = raw.trim();
    if target.is_empty() {
        return Err(AssignmentError::EmptyName);
    }

    let Some(open_idx) = target.find('(') else {
        validate_identifier(context, target, target)?;
        return Ok(AssignmentTarget::Variable {
            name: target.to_string(),
        });
    };

    if !target.ends_with(')') || target[open_idx + 1..target.len() - 1].contains('(') {
        return Err(AssignmentError::Parse(format!(
            "invalid function assignment target '{}'",
            target
        )));
    }

    let name = target[..open_idx].trim();
    validate_identifier(context, name, name)?;

    let inner = &target[open_idx + 1..target.len() - 1];
    let mut params = Vec::new();
    let mut seen = std::collections::HashSet::new();
    if !inner.trim().is_empty() {
        for param in inner.split(',') {
            let param = param.trim();
            validate_identifier(context, param, param)?;
            if !seen.insert(param.to_string()) {
                return Err(AssignmentError::Parse(format!(
                    "duplicate parameter '{}' in function assignment",
                    param
                )));
            }
            params.push(param.to_string());
        }
    }

    Ok(AssignmentTarget::Function {
        name: name.to_string(),
        params,
    })
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
    let target = parse_assignment_target(context, name)?;

    let rhs_expr = cas_parser::parse(expr_str, &mut simplifier.context)
        .map_err(|e| AssignmentError::Parse(e.to_string()))?;

    match target {
        AssignmentTarget::Variable { name } => {
            context.assignment_unset_binding(&name);
            context.assignment_unset_function(&name);

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

            context.assignment_set_binding(name, result);
            Ok(result)
        }
        AssignmentTarget::Function { name, params } => {
            context.assignment_unset_binding(&name);
            context.assignment_unset_function(&name);

            let rhs_session_resolved =
                match context.assignment_resolve_session_refs(&mut simplifier.context, rhs_expr) {
                    Ok(resolved) => resolved,
                    Err(_) => rhs_expr,
                };
            let mut shadow: Vec<&str> = params.iter().map(String::as_str).collect();
            shadow.push(name.as_str());
            let rhs_substituted = context.assignment_substitute_bindings_with_shadow(
                &mut simplifier.context,
                rhs_session_resolved,
                &shadow,
            );

            let result = if lazy {
                rhs_substituted
            } else {
                let (simplified, _steps) = simplifier.simplify(rhs_substituted);
                unwrap_hold_top(&simplifier.context, simplified)
            };

            context.assignment_set_function(name, params, result);
            Ok(result)
        }
    }
}
