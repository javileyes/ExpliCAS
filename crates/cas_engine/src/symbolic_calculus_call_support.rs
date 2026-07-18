//! Parsing/render helpers for `integrate(...)` and `diff(...)` call forms.

use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedVarCall {
    pub target: ExprId,
    pub var_name: String,
}

/// Parse `integrate(target, var)` and `integrate(target)` (defaults to `x`).
pub struct DefiniteIntegralCall {
    pub target: ExprId,
    pub var_expr: ExprId,
    pub var_name: String,
    pub lower: ExprId,
    pub upper: ExprId,
}

/// integrate(f, x, a, b): the definite-integral call shape (mirror of the
/// finite-aggregate sum/product extractor).
pub(crate) fn try_extract_definite_integrate_call(
    ctx: &Context,
    expr: ExprId,
) -> Option<DefiniteIntegralCall> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 4 {
        return None;
    }
    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return None;
    };
    Some(DefiniteIntegralCall {
        target: args[0],
        var_expr: args[1],
        var_name: ctx.sym_name(*var_sym).to_string(),
        lower: args[2],
        upper: args[3],
    })
}

pub(crate) fn try_extract_integrate_call(ctx: &Context, expr: ExprId) -> Option<NamedVarCall> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*fn_id) != "integrate" {
        return None;
    }

    if args.len() == 2 {
        let target = args[0];
        let var_expr = args[1];
        let Expr::Variable(var_sym) = ctx.get(var_expr) else {
            return None;
        };
        return Some(NamedVarCall {
            target,
            var_name: ctx.sym_name(*var_sym).to_string(),
        });
    }

    if args.len() == 1 {
        return Some(NamedVarCall {
            target: args[0],
            var_name: "x".to_string(),
        });
    }

    None
}

/// A vectorial-verb call `verbo(field, [v1, v2, …])` (Fase 2 V3+).
pub(crate) struct FieldVarsCall {
    pub target: ExprId,
    pub var_names: Vec<String>,
}

/// Parse `name(field, [vars])` for the vectorial verbs: EXACT arity 2, and
/// `args[1]` an n×1 or 1×n `Matrix` whose entries are PURE `Expr::Variable`s
/// (`Matrix::from_expr` accepts anything, so the shape validation lives here).
/// Anything else returns `None` and the call stays an honest residual. The
/// list-of-vars arity is EXCLUSIVE to the verbs — `diff`'s 3+-arity SymPy
/// convention (`diff(f, x, y)` = mixed partial) is a different owner and is
/// never rerouted here.
pub(crate) fn try_extract_field_vars_call(
    ctx: &Context,
    expr: ExprId,
    names: &[&str],
) -> Option<FieldVarsCall> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !names.contains(&ctx.sym_name(*fn_id)) {
        return None;
    }
    if args.len() != 2 {
        return None;
    }
    let target = args[0];
    let Expr::Matrix { rows, cols, data } = ctx.get(args[1]) else {
        return None;
    };
    if (*rows != 1 && *cols != 1) || data.is_empty() {
        return None;
    }
    let mut var_names = Vec::with_capacity(data.len());
    for &v in data {
        let Expr::Variable(sym) = ctx.get(v) else {
            return None;
        };
        var_names.push(ctx.sym_name(*sym).to_string());
    }
    Some(FieldVarsCall { target, var_names })
}

/// Parse `diff(target, var)` with explicit variable.
pub(crate) fn try_extract_diff_call(ctx: &Context, expr: ExprId) -> Option<NamedVarCall> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*fn_id) != "diff" || args.len() != 2 {
        return None;
    }

    let target = args[0];
    let var_expr = args[1];
    let Expr::Variable(var_sym) = ctx.get(var_expr) else {
        return None;
    };
    Some(NamedVarCall {
        target,
        var_name: ctx.sym_name(*var_sym).to_string(),
    })
}

/// Desugar a higher-order / mixed-partial diff call into nested 2-arg `diff(...)`.
///
/// Accepts `diff(expr, v1, [n1,] v2, [n2,] ...)` (3+ args) where each `vi` is a
/// differentiation variable and each optional integer `ni ≥ 1` is a repeat count on
/// the *preceding* variable — the SymPy convention: `diff(f, x, 2)` is the second
/// derivative w.r.t. `x`, `diff(f, x, y)` is the mixed partial ∂²f/∂y∂x, and
/// `diff(f, x, 2, y)` mixes both. Returns the equivalent nested two-argument diff
/// expression so the ordinary `DiffRule` cascade evaluates each layer unchanged.
///
/// Returns `None` when the call is not a 3+-arg `diff`, when a count is not a
/// positive integer literal, or when an integer would lead the variable list (a
/// count needs a variable to repeat). The malformed call is then left untouched.
pub(crate) fn try_desugar_higher_order_diff(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*fn_id) != "diff" || args.len() < 3 {
        return None;
    }
    let fn_id = *fn_id;
    let args = args.clone();
    let target = args[0];

    // SYMBOLIC ORDER vs mixed partial: `diff(f, x, n)` is AMBIGUOUS when the trailing
    // token `n` is a bare symbol NOT occurring free in `f` — it could be a symbolic
    // differentiation ORDER (unsupported) or a phantom second variable. Treating it as
    // a variable computes `∂/∂n ∂/∂x f = 0` (`diff(e^x,x,n) → 0`, a concrete wrong
    // value). Decline (leave untouched → honest echo), exactly as a non-positive
    // integer count already does. A genuine mixed partial (`diff(x^3·y^2, x, y)`, `y`
    // free in the target) still desugars. Scoped to the 3-arg shape so multi-variable
    // and mixed-count desugars are untouched.
    if args.len() == 3 {
        if let Expr::Variable(_) = ctx.get(args[2]) {
            let free = cas_ast::traversal::collect_variables(ctx, target);
            let Expr::Variable(sym) = ctx.get(args[2]) else {
                unreachable!()
            };
            if !free.contains(ctx.sym_name(*sym)) {
                return None;
            }
        }
    }

    // Flatten the `(variable (count)?)+` tail into an ordered list of single
    // differentiation steps, expanding each integer count on its preceding variable.
    let mut steps: Vec<ExprId> = Vec::new();
    let mut last_var: Option<ExprId> = None;
    for &tok in &args[1..] {
        if matches!(ctx.get(tok), Expr::Variable(_)) {
            steps.push(tok);
            last_var = Some(tok);
        } else {
            // A bare token must be a positive-integer repeat count on the last variable;
            // the variable was already pushed once, so add `n - 1` more applications.
            let n = cas_math::numeric::as_i64(ctx, tok)?;
            if n < 1 {
                return None;
            }
            let repeated = last_var?;
            for _ in 1..n {
                steps.push(repeated);
            }
        }
    }

    // Wrap the target in `diff(·, var)` once per step, innermost variable first.
    let mut current = target;
    for var in steps {
        current = ctx.add(Expr::Function(fn_id, vec![current, var]));
    }
    Some(current)
}

/// Render `integrate(target, var)` description using a caller-provided expression renderer.
pub(crate) fn render_integrate_desc_with<F>(call: &NamedVarCall, mut render_expr: F) -> String
where
    F: FnMut(ExprId) -> String,
{
    format!("integrate({}, {})", render_expr(call.target), call.var_name)
}

/// Render `diff(target, var)` description using a caller-provided expression renderer.
pub(crate) fn render_diff_desc_with<F>(call: &NamedVarCall, mut render_expr: F) -> String
where
    F: FnMut(ExprId) -> String,
{
    format!("diff({}, {})", render_expr(call.target), call.var_name)
}

#[cfg(test)]
#[path = "symbolic_calculus_call_support_tests.rs"]
mod tests;
