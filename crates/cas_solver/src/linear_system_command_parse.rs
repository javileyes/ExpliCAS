mod equations;
mod invocation;
mod split;
mod vars;

use cas_ast::Context;
use cas_ast::{ExprId, RelOp};

#[derive(Debug, Clone)]
pub(crate) struct LinearSystemSpec {
    pub(crate) exprs: Vec<ExprId>,
    pub(crate) vars: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LinearSystemSpecError {
    InvalidPartCount,
    InvalidVariableName { name: String },
    ParseEquation { position: usize, message: String },
    ExpectedEquation { position: usize, input: String },
    UnsupportedRelation,
}

pub(crate) fn ensure_equation_relation(op: RelOp) -> Result<(), LinearSystemSpecError> {
    if op == RelOp::Eq {
        Ok(())
    } else {
        Err(LinearSystemSpecError::UnsupportedRelation)
    }
}

pub(crate) fn parse_linear_system_spec(
    ctx: &mut Context,
    input: &str,
) -> Result<LinearSystemSpec, LinearSystemSpecError> {
    let parts = split::split_linear_system_parts(input);

    if parts.len() < 4 || !parts.len().is_multiple_of(2) {
        return Err(LinearSystemSpecError::InvalidPartCount);
    }

    let n = parts.len() / 2;
    let eq_parts = &parts[0..n];
    let var_parts = &parts[n..2 * n];
    let vars = vars::parse_linear_system_vars(var_parts)?;
    let exprs = equations::parse_linear_system_exprs(ctx, eq_parts)?;
    // V7d (cierre vectorial, decisión del usuario 2026-07-18): pre-evaluate inline
    // `diff(f, x)` calls BEFORE the multipoly conversion — the critical-points flow
    // `solve([diff(f,x)=0, diff(f,y)=0], [x,y])` otherwise dies at "expression is
    // not a polynomial over Q". A diff the support differentiator declines stays in
    // place and fails conversion exactly as today (honest). Non-linear gradients
    // keep declining downstream (scope-out).
    let exprs = exprs
        .into_iter()
        .map(|e| pre_evaluate_inline_diff_calls(ctx, e))
        .collect();
    // Frente S: "la lista de incógnitas manda" aplicado a los NOMBRES — una
    // constante nombrada (e, pi, phi) DECLARADA como incógnita es una
    // variable dentro del canal de sistemas (espejo de la decisión D14 de
    // dsolve: la excepción vive donde vive el contexto; el significado
    // global de `e` no se toca). `i` queda fuera a propósito: la unidad
    // imaginaria es estructural para la maquinaria compleja — declinar es
    // más seguro que sombrearla.
    let exprs = shadow_declared_constant_names(ctx, exprs, &vars);

    Ok(LinearSystemSpec { exprs, vars })
}

/// Fold every numerically-closed subtree to a single exact `Number` node
/// (`2-1` → `1`, `6-3·2` → `0`) via `as_rational_const` — the raw support
/// derivative keeps power-rule artifacts whose non-literal exponents the
/// multipoly conversion rejects (memoria: `numeric_value` solo casa literales).
fn fold_numeric_subtrees(ctx: &mut Context, expr: ExprId) -> ExprId {
    use cas_ast::Expr;
    if !matches!(ctx.get(expr), Expr::Number(_)) {
        if let Some(q) = cas_math::numeric_eval::as_rational_const(ctx, expr) {
            return ctx.add(Expr::Number(q));
        }
    }
    match ctx.get(expr).clone() {
        Expr::Add(a, b) => {
            let (a, b) = (fold_numeric_subtrees(ctx, a), fold_numeric_subtrees(ctx, b));
            ctx.add(Expr::Add(a, b))
        }
        Expr::Sub(a, b) => {
            let (a, b) = (fold_numeric_subtrees(ctx, a), fold_numeric_subtrees(ctx, b));
            ctx.add(Expr::Sub(a, b))
        }
        Expr::Mul(a, b) => {
            let (a, b) = (fold_numeric_subtrees(ctx, a), fold_numeric_subtrees(ctx, b));
            ctx.add(Expr::Mul(a, b))
        }
        Expr::Div(a, b) => {
            let (a, b) = (fold_numeric_subtrees(ctx, a), fold_numeric_subtrees(ctx, b));
            ctx.add(Expr::Div(a, b))
        }
        Expr::Pow(a, b) => {
            let (a, b) = (fold_numeric_subtrees(ctx, a), fold_numeric_subtrees(ctx, b));
            ctx.add(Expr::Pow(a, b))
        }
        Expr::Neg(inner) => {
            let inner = fold_numeric_subtrees(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Function(fn_id, args) => {
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|&a| fold_numeric_subtrees(ctx, a))
                .collect();
            ctx.add(Expr::Function(fn_id, new_args))
        }
        _ => expr,
    }
}

/// Bottom-up rewrite of every 2-arg `diff(target, var)` node into its computed
/// derivative via the shared support differentiator (no Simplifier available on
/// this path — and none needed: the derivative machinery is a pure function).
/// 3+-arg diff (higher-order/mixed) stays untouched — a named residual.
fn pre_evaluate_inline_diff_calls(ctx: &mut Context, expr: ExprId) -> ExprId {
    use cas_ast::Expr;
    use cas_math::symbolic_differentiation_support::differentiate_symbolic_expr;
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|&a| pre_evaluate_inline_diff_calls(ctx, a))
                .collect();
            let fn_name = ctx.sym_name(fn_id).to_string();
            if fn_name == "diff" && new_args.len() == 2 {
                if let Expr::Variable(v) = ctx.get(new_args[1]) {
                    let var_name = ctx.sym_name(*v).to_string();
                    if let Some(derived) = differentiate_symbolic_expr(ctx, new_args[0], &var_name)
                    {
                        // The raw derivative carries power-rule artifacts (`x^(2-1)`)
                        // whose non-literal exponents the multipoly conversion rejects —
                        // fold every numerically-closed subtree to its exact rational.
                        return fold_numeric_subtrees(ctx, derived);
                    }
                }
            }
            ctx.add(Expr::Function(fn_id, new_args))
        }
        Expr::Add(a, b) => {
            let (a, b) = (
                pre_evaluate_inline_diff_calls(ctx, a),
                pre_evaluate_inline_diff_calls(ctx, b),
            );
            ctx.add(Expr::Add(a, b))
        }
        Expr::Sub(a, b) => {
            let (a, b) = (
                pre_evaluate_inline_diff_calls(ctx, a),
                pre_evaluate_inline_diff_calls(ctx, b),
            );
            ctx.add(Expr::Sub(a, b))
        }
        Expr::Mul(a, b) => {
            let (a, b) = (
                pre_evaluate_inline_diff_calls(ctx, a),
                pre_evaluate_inline_diff_calls(ctx, b),
            );
            ctx.add(Expr::Mul(a, b))
        }
        Expr::Div(a, b) => {
            let (a, b) = (
                pre_evaluate_inline_diff_calls(ctx, a),
                pre_evaluate_inline_diff_calls(ctx, b),
            );
            ctx.add(Expr::Div(a, b))
        }
        Expr::Pow(a, b) => {
            let (a, b) = (
                pre_evaluate_inline_diff_calls(ctx, a),
                pre_evaluate_inline_diff_calls(ctx, b),
            );
            ctx.add(Expr::Pow(a, b))
        }
        Expr::Neg(inner) => {
            let inner = pre_evaluate_inline_diff_calls(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        _ => expr,
    }
}

pub(crate) fn parse_linear_system_invocation_input(line: &str) -> String {
    invocation::parse_linear_system_invocation_input(line)
}

/// Replace named-constant nodes (`e`, `pi`, `phi`) by plain variables when
/// the SAME name was declared in the unknowns list. Scoped to this channel:
/// outside the declared list the constants keep their global meaning
/// (`x + e·y = 1` still solves with Euler's e as an exact coefficient).
fn shadow_declared_constant_names(
    ctx: &mut Context,
    exprs: Vec<cas_ast::ExprId>,
    vars: &[String],
) -> Vec<cas_ast::ExprId> {
    use cas_ast::{Constant, Expr};
    let shadowable: Vec<(cas_ast::ExprId, cas_ast::ExprId)> = vars
        .iter()
        .filter_map(|name| {
            let constant = match name.as_str() {
                "e" => Constant::E,
                "pi" => Constant::Pi,
                "phi" => Constant::Phi,
                _ => return None,
            };
            let target = ctx.add(Expr::Constant(constant));
            let replacement = ctx.var(name);
            Some((target, replacement))
        })
        .collect();
    if shadowable.is_empty() {
        return exprs;
    }
    exprs
        .into_iter()
        .map(|expr| {
            shadowable.iter().fold(expr, |acc, &(target, replacement)| {
                crate::substitute::substitute_power_aware(
                    ctx,
                    acc,
                    target,
                    replacement,
                    crate::substitute::SubstituteOptions::exact(),
                )
            })
        })
        .collect()
}
