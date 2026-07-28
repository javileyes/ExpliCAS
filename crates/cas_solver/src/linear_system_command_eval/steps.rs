//! Educational narration for system solves (frente S · S3).
//!
//! Every outcome family narrates: identification, method, and the honesty
//! clause (verification / validity condition / rank story). Descriptions are
//! Spanish canonical templates localized es/en at the presentation boundary
//! (the SolveDesc table), matching the dsolve D13 mould.

use cas_ast::{Context, Equation, ExprId, RelOp};

use super::nonlinear::SystemNarration;

/// Substitution that rebuilds RAW, for narration only.
///
/// The canonical `substitute_expr_by_id` re-adds every node through
/// `Context::add`, which reorders and folds: `2·x + 3·y` with `x = 3, y = 2`
/// came out as `2·3 + 2·3` — two identical-looking terms where the reader has
/// to trace which value replaced which unknown. A verification step that
/// cannot be read back against its own equation is the same filler the old
/// narration was, so the substituted tree keeps the shape of the original.
///
/// Arithmetic nodes are rebuilt raw; anything else (functions, matrices)
/// delegates to the canonical substitution — correct, just canonical, and a
/// linear system does not produce those shapes anyway.
fn substitute_raw(ctx: &mut Context, root: ExprId, bindings: &[(ExprId, ExprId)]) -> ExprId {
    if let Some((_, replacement)) = bindings.iter().find(|(target, _)| *target == root) {
        return *replacement;
    }
    let expr = ctx.get(root).clone();
    macro_rules! binary {
        ($variant:ident, $l:expr, $r:expr) => {{
            let l = substitute_raw(ctx, $l, bindings);
            let r = substitute_raw(ctx, $r, bindings);
            ctx.add_raw(cas_ast::Expr::$variant(l, r))
        }};
    }
    match expr {
        cas_ast::Expr::Add(l, r) => binary!(Add, l, r),
        cas_ast::Expr::Sub(l, r) => binary!(Sub, l, r),
        cas_ast::Expr::Mul(l, r) => binary!(Mul, l, r),
        cas_ast::Expr::Div(l, r) => binary!(Div, l, r),
        cas_ast::Expr::Pow(l, r) => binary!(Pow, l, r),
        cas_ast::Expr::Neg(inner) => {
            let inner = substitute_raw(ctx, inner, bindings);
            ctx.add_raw(cas_ast::Expr::Neg(inner))
        }
        cas_ast::Expr::Number(_) | cas_ast::Expr::Constant(_) | cas_ast::Expr::Variable(_) => root,
        _ => {
            let mut out = root;
            for (target, replacement) in bindings {
                out = cas_ast::substitute_expr_by_id(ctx, out, *target, *replacement);
            }
            out
        }
    }
}

fn eq_zero(ctx: &mut Context, lhs: ExprId) -> Equation {
    let zero = ctx.num(0);
    Equation {
        lhs,
        rhs: zero,
        op: RelOp::Eq,
    }
}

pub(super) fn build_system_solve_steps(
    ctx: &mut Context,
    exprs: &[ExprId],
    vars: &[String],
    result: &crate::LinSolveResult,
    nonlinear: Option<&SystemNarration>,
) -> Vec<crate::SolveStep> {
    let mut steps: Vec<crate::SolveStep> = Vec::new();
    let anchor = exprs.first().copied();
    let Some(anchor) = anchor else {
        return steps;
    };
    let identify_eq = eq_zero(ctx, anchor);

    // Identification, ONE step per equation. The single header step used
    // `exprs.first()` as a filler anchor, so a two-equation system announced
    // itself over equation 1 and equation 2 never appeared in the trace at
    // all — and the steps that followed repeated that same snapshot while
    // claiming to solve and to verify the whole system.
    for (index, &expr) in exprs.iter().enumerate() {
        let eq = eq_zero(ctx, expr);
        steps.push(crate::SolveStep::new(
            format!(
                "Ecuación {} de {} del sistema en las incógnitas [{}]",
                index + 1,
                exprs.len(),
                vars.join(", ")
            ),
            eq,
            crate::ImportanceLevel::High,
        ));
    }

    match result {
        crate::LinSolveResult::Unique(values) => {
            // The method named as it RUNS: `solve_nxn_gauss` reduces the
            // augmented matrix to row echelon over ℚ and back-substitutes.
            // The old text hedged «Cramer/Gauss» over a filler snapshot; each
            // value now rides its own equation, which is what back-substitution
            // actually produces.
            for (var, value) in vars.iter().zip(values.iter()) {
                let var_expr = ctx.var(var);
                let value_expr = ctx.add(cas_ast::Expr::Number(value.clone()));
                steps.push(crate::SolveStep::new(
                    format!(
                        "Eliminación gaussiana exacta sobre racionales y back-substitución: {var}"
                    ),
                    Equation {
                        lhs: var_expr,
                        rhs: value_expr,
                        op: RelOp::Eq,
                    },
                    crate::ImportanceLevel::High,
                ));
            }
            // The honesty clause, made visible: the old step ASSERTED that
            // every value substitutes exactly and showed equation 1 untouched.
            // Now each original equation is shown WITH the values in place, so
            // the reader can do the arithmetic that backs the claim.
            let bindings: Vec<(ExprId, ExprId)> = vars
                .iter()
                .zip(values.iter())
                .map(|(var, value)| {
                    let var_expr = ctx.var(var);
                    let value_expr = ctx.add(cas_ast::Expr::Number(value.clone()));
                    (var_expr, value_expr)
                })
                .collect();
            for (index, &expr) in exprs.iter().enumerate() {
                let substituted = substitute_raw(ctx, expr, &bindings);
                let eq = eq_zero(ctx, substituted);
                steps.push(crate::SolveStep::new(
                    format!(
                        "Verificación exacta: sustituir los valores en la ecuación {}",
                        index + 1
                    ),
                    eq,
                    crate::ImportanceLevel::Medium,
                ));
            }
        }
        crate::LinSolveResult::UniqueExpr {
            values,
            nonzero_conditions,
        } => {
            steps.push(crate::SolveStep::new(
                "Coeficientes simbólicos: la lista de incógnitas decide la linealidad y los parámetros van a los coeficientes",
                identify_eq.clone(),
                crate::ImportanceLevel::High,
            ));
            // Cramer PRODUCE un valor por incógnita, así que eso es lo que
            // muestra — un solo paso con la ecuación 1 repetida anunciaba el
            // método sobre algo que no estaba resolviendo (el segundo
            // superviviente D5b). Mismo trato que el brazo `Unique`.
            for (var, &value) in vars.iter().zip(values.iter()) {
                let var_expr = ctx.var(var);
                steps.push(crate::SolveStep::new(
                    format!("Resolver por Cramer exacto sobre polinomios en los parámetros: {var}"),
                    Equation {
                        lhs: var_expr,
                        rhs: value,
                        op: RelOp::Eq,
                    },
                    crate::ImportanceLevel::High,
                ));
            }
            for &cond in nonzero_conditions {
                let cond_eq = Equation {
                    lhs: cond,
                    rhs: identify_eq.rhs,
                    op: RelOp::Neq,
                };
                steps.push(crate::SolveStep::new(
                    "Condición de validez: el determinante debe ser distinto de cero",
                    cond_eq,
                    crate::ImportanceLevel::High,
                ));
            }
        }
        crate::LinSolveResult::SolutionPairs(pairs) => {
            if let Some(SystemNarration::Resultant(narr)) = nonlinear {
                let res_eq = eq_zero(ctx, narr.resultant);
                steps.push(crate::SolveStep::new(
                    format!(
                        "Eliminar {} por la resultante de Sylvester (determinante {}×{})",
                        narr.eliminated_var, narr.sylvester_dim, narr.sylvester_dim
                    ),
                    res_eq.clone(),
                    crate::ImportanceLevel::High,
                ));
                steps.push(crate::SolveStep::new(
                    format!("Resolver la resultante univariable en {}", narr.root_var),
                    res_eq,
                    crate::ImportanceLevel::High,
                ));
                steps.push(crate::SolveStep::new(
                    format!(
                        "Back-substitute por raíz y verificar cada par contra AMBAS ecuaciones: {} pares verificados emitidos",
                        pairs.len()
                    ),
                    identify_eq.clone(),
                    crate::ImportanceLevel::Medium,
                ));
            }
            if let Some(SystemNarration::Substitution(narr)) = nonlinear {
                let u_var = ctx.var(&narr.isolated_var);
                let iso_eq = Equation {
                    lhs: u_var,
                    rhs: narr.isolation,
                    op: RelOp::Eq,
                };
                steps.push(crate::SolveStep::new(
                    format!(
                        "Aislar {} de la ecuación {} (lineal en esa incógnita)",
                        narr.isolated_var,
                        narr.source_index + 1
                    ),
                    iso_eq,
                    crate::ImportanceLevel::High,
                ));
                let univariate_eq = eq_zero(ctx, narr.univariate);
                steps.push(crate::SolveStep::new(
                    format!(
                        "Sustituir en la otra ecuación y resolver la univariable en {}",
                        narr.free_var
                    ),
                    univariate_eq,
                    crate::ImportanceLevel::High,
                ));
                steps.push(crate::SolveStep::new(
                    format!(
                        "Verificar cada par contra AMBAS ecuaciones originales: {} pares verificados emitidos",
                        pairs.len()
                    ),
                    identify_eq,
                    crate::ImportanceLevel::Medium,
                ));
            }
        }
        crate::LinSolveResult::Infinite => {
            steps.push(crate::SolveStep::new(
                "Las ecuaciones son dependientes (rango menor que el número de incógnitas): infinitas soluciones",
                identify_eq,
                crate::ImportanceLevel::High,
            ));
        }
        crate::LinSolveResult::Inconsistent => {
            steps.push(crate::SolveStep::new(
                "Las ecuaciones son inconsistentes (ninguna asignación satisface todas): sin solución",
                identify_eq,
                crate::ImportanceLevel::High,
            ));
        }
    }
    steps
}
