//! Educational narration for system solves (frente S · S3).
//!
//! Every outcome family narrates: identification, method, and the honesty
//! clause (verification / validity condition / rank story). Descriptions are
//! Spanish canonical templates localized es/en at the presentation boundary
//! (the SolveDesc table), matching the dsolve D13 mould.

use cas_ast::{Context, Equation, ExprId, RelOp};

use super::nonlinear::SystemNarration;

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
    steps.push(crate::SolveStep::new(
        format!(
            "Identificar sistema de {} ecuaciones con incógnitas [{}]",
            exprs.len(),
            vars.join(", ")
        ),
        identify_eq.clone(),
        crate::ImportanceLevel::High,
    ));

    match result {
        crate::LinSolveResult::Unique(_) => {
            steps.push(crate::SolveStep::new(
                "Resolver el sistema lineal por eliminación exacta (Cramer/Gauss sobre racionales)",
                identify_eq.clone(),
                crate::ImportanceLevel::High,
            ));
            steps.push(crate::SolveStep::new(
                "Solución única: cada valor sustituye exacto en todas las ecuaciones",
                identify_eq,
                crate::ImportanceLevel::Medium,
            ));
        }
        crate::LinSolveResult::UniqueExpr {
            nonzero_conditions, ..
        } => {
            steps.push(crate::SolveStep::new(
                "Coeficientes simbólicos: la lista de incógnitas decide la linealidad y los parámetros van a los coeficientes",
                identify_eq.clone(),
                crate::ImportanceLevel::High,
            ));
            steps.push(crate::SolveStep::new(
                "Resolver por Cramer exacto sobre polinomios en los parámetros",
                identify_eq.clone(),
                crate::ImportanceLevel::High,
            ));
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
