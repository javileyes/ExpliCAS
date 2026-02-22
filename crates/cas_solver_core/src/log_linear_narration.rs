use cas_ast::{BuiltinFn, Context, Equation, Expr, ExprId};
use cas_math::expr_predicates::is_one_expr as is_one;

use crate::isolation_utils::contains_var;

/// Generic narration step payload (solver-core does not depend on engine step types).
#[derive(Debug, Clone)]
pub struct LogLinearNarrationStep {
    pub description: String,
    pub equation_after: Equation,
}

/// Marker used by solver traces to indicate log-linear rewrite entrypoint.
pub const TAKE_LOG_BOTH_SIDES_STEP: &str = "Take log base e of both sides";

/// Compact narration for the log-linear collect/factor step.
pub fn collect_and_factor_terms_message(var: &str) -> String {
    format!("Collect and factor {} terms", var)
}

/// Check whether a single step description is the log-linear entry marker.
pub fn is_log_linear_take_log_step(description: &str) -> bool {
    description == TAKE_LOG_BOTH_SIDES_STEP
}

/// Check if a step stream starts with the log-linear marker.
pub fn is_log_linear_pattern(steps: &[LogLinearNarrationStep]) -> bool {
    steps
        .first()
        .map(|s| is_log_linear_take_log_step(&s.description))
        .unwrap_or(false)
}

/// Strip identity multipliers (`1*expr`/`expr*1`) recursively for cleaner display.
pub fn strip_mul_one(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    match expr_data {
        Expr::Mul(l, r) => {
            let clean_l = strip_mul_one(ctx, l);
            let clean_r = strip_mul_one(ctx, r);

            if is_one(ctx, clean_l) {
                return clean_r;
            }
            if is_one(ctx, clean_r) {
                return clean_l;
            }

            ctx.add(Expr::Mul(clean_l, clean_r))
        }
        Expr::Add(l, r) => {
            let clean_l = strip_mul_one(ctx, l);
            let clean_r = strip_mul_one(ctx, r);
            ctx.add(Expr::Add(clean_l, clean_r))
        }
        Expr::Sub(l, r) => {
            let clean_l = strip_mul_one(ctx, l);
            let clean_r = strip_mul_one(ctx, r);
            ctx.add(Expr::Sub(clean_l, clean_r))
        }
        Expr::Neg(inner) => {
            let clean_inner = strip_mul_one(ctx, inner);
            ctx.add(Expr::Neg(clean_inner))
        }
        Expr::Div(l, r) => {
            let clean_l = strip_mul_one(ctx, l);
            let clean_r = strip_mul_one(ctx, r);
            ctx.add(Expr::Div(clean_l, clean_r))
        }
        Expr::Pow(base, exp) => {
            let clean_base = strip_mul_one(ctx, base);
            let clean_exp = strip_mul_one(ctx, exp);
            ctx.add(Expr::Pow(clean_base, clean_exp))
        }
        Expr::Function(fn_id, args) => {
            let mut changed = false;
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|&a| {
                    let na = strip_mul_one(ctx, a);
                    if na != a {
                        changed = true;
                    }
                    na
                })
                .collect();
            if changed {
                ctx.add(Expr::Function(fn_id, new_args))
            } else {
                expr
            }
        }
        Expr::Hold(inner) => {
            let clean = strip_mul_one(ctx, inner);
            if clean != inner {
                ctx.add(Expr::Hold(clean))
            } else {
                expr
            }
        }
        Expr::Matrix { rows, cols, data } => {
            let mut changed = false;
            let new_data: Vec<ExprId> = data
                .iter()
                .map(|&d| {
                    let nd = strip_mul_one(ctx, d);
                    if nd != d {
                        changed = true;
                    }
                    nd
                })
                .collect();
            if changed {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                })
            } else {
                expr
            }
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => expr,
    }
}

/// Apply `strip_mul_one` to both sides of an equation.
pub fn strip_equation_mul_one(ctx: &mut Context, eq: &Equation) -> Equation {
    Equation {
        lhs: strip_mul_one(ctx, eq.lhs),
        rhs: strip_mul_one(ctx, eq.rhs),
        op: eq.op.clone(),
    }
}

/// Expand one distributive pattern `k*(a+b)` or `(a+b)*k`.
pub fn try_expand_distributive(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Mul(l, r) = ctx.get(expr).clone() {
        if let Expr::Add(a, b) = ctx.get(r).clone() {
            let term1 = ctx.add(Expr::Mul(l, a));
            let term2 = ctx.add(Expr::Mul(l, b));
            return Some(ctx.add(Expr::Add(term1, term2)));
        }
        if let Expr::Add(a, b) = ctx.get(l).clone() {
            let term1 = ctx.add(Expr::Mul(a, r));
            let term2 = ctx.add(Expr::Mul(b, r));
            return Some(ctx.add(Expr::Add(term1, term2)));
        }
    }
    None
}

/// Extract `(constant_term, var_term)` from `const + var_term`.
pub fn try_extract_constant_and_var_term(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    if let Expr::Add(l, r) = ctx.get(expr) {
        let l_has_var = contains_var(ctx, *l, var);
        let r_has_var = contains_var(ctx, *r, var);

        match (l_has_var, r_has_var) {
            (false, true) => Some((*l, *r)),
            (true, false) => Some((*r, *l)),
            _ => None,
        }
    } else {
        None
    }
}

/// Factor a variable from `var*a ± var*b` into `var*(a ± b)`.
pub fn try_factor_variable(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Sub(l, r) => {
            let l_coef = try_extract_var_coefficient(ctx, l, var)?;
            let r_coef = try_extract_var_coefficient(ctx, r, var)?;
            let coef_diff = ctx.add(Expr::Sub(l_coef, r_coef));
            let var_id = ctx.var(var);
            Some(ctx.add(Expr::Mul(var_id, coef_diff)))
        }
        Expr::Add(l, r) => {
            let l_coef = try_extract_var_coefficient(ctx, l, var)?;
            let r_coef = try_extract_var_coefficient(ctx, r, var)?;
            let coef_sum = ctx.add(Expr::Add(l_coef, r_coef));
            let var_id = ctx.var(var);
            Some(ctx.add(Expr::Mul(var_id, coef_sum)))
        }
        _ => None,
    }
}

fn try_extract_var_coefficient(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    if let Expr::Mul(l, r) = ctx.get(expr) {
        if matches!(ctx.get(*l), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var) {
            return Some(*r);
        }
        if matches!(ctx.get(*r), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var) {
            return Some(*l);
        }
    }
    None
}

/// Rewrite `ln(a^b)` as `b*ln(a)` for didactic display.
pub fn try_rewrite_ln_power(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Function(fn_id, args) if ctx.is_builtin(fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            let inner = args[0];
            if let Expr::Pow(base, exp) = ctx.get(inner).clone() {
                let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                return Some(ctx.add(Expr::Mul(exp, ln_base)));
            }
            None
        }
        _ => None,
    }
}

/// Build detailed didactic sub-steps for log-linear collection (`Collect terms in x`).
///
/// The returned steps contain only description + equation; caller decides rendering
/// concerns such as importance/substeps wrappers.
pub fn build_detailed_collect_steps(
    ctx: &mut Context,
    log_eq_opt: Option<&Equation>,
    final_eq: &Equation,
    var: &str,
) -> Vec<LogLinearNarrationStep> {
    let mut steps = Vec::new();

    // If we don't have the equation right after taking logs, keep compact fallback.
    let log_eq = match log_eq_opt {
        Some(eq) => eq,
        None => {
            steps.push(LogLinearNarrationStep {
                description: collect_and_factor_terms_message(var),
                equation_after: final_eq.clone(),
            });
            return steps;
        }
    };

    if let Some(expanded_lhs) = try_expand_distributive(ctx, log_eq.lhs) {
        let expand_eq = Equation {
            lhs: expanded_lhs,
            rhs: log_eq.rhs,
            op: log_eq.op.clone(),
        };
        let clean_expand_eq = strip_equation_mul_one(ctx, &expand_eq);
        steps.push(LogLinearNarrationStep {
            description: "Expand distributive law".to_string(),
            equation_after: clean_expand_eq,
        });

        if let Some((constant, var_term_lhs)) =
            try_extract_constant_and_var_term(ctx, expanded_lhs, var)
        {
            let var_term_rhs = log_eq.rhs;
            let moved_rhs = ctx.add(Expr::Sub(var_term_rhs, var_term_lhs));
            let move_eq = Equation {
                lhs: constant,
                rhs: moved_rhs,
                op: log_eq.op.clone(),
            };
            let clean_move_eq = strip_equation_mul_one(ctx, &move_eq);
            steps.push(LogLinearNarrationStep {
                description: format!("Move {} terms to one side", var),
                equation_after: clean_move_eq,
            });

            if let Some(factored_rhs) = try_factor_variable(ctx, moved_rhs, var) {
                let factor_eq = Equation {
                    lhs: constant,
                    rhs: factored_rhs,
                    op: log_eq.op.clone(),
                };
                let clean_factor_eq = strip_equation_mul_one(ctx, &factor_eq);
                steps.push(LogLinearNarrationStep {
                    description: format!("Factor out {}", var),
                    equation_after: clean_factor_eq,
                });
            }
        }
    }

    if steps.is_empty() {
        steps.push(LogLinearNarrationStep {
            description: collect_and_factor_terms_message(var),
            equation_after: final_eq.clone(),
        });
    }

    steps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_mul_one_removes_identity_factors() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let expr = ctx.add(Expr::Mul(one, x));
        assert_eq!(strip_mul_one(&mut ctx, expr), x);
    }

    #[test]
    fn extract_constant_and_var_term_detects_linear_sum() {
        let mut ctx = Context::new();
        let c = ctx.var("c");
        let x = ctx.var("x");
        let k = ctx.var("k");
        let xk = ctx.add(Expr::Mul(x, k));
        let expr = ctx.add(Expr::Add(c, xk));

        let (constant, with_var) =
            try_extract_constant_and_var_term(&ctx, expr, "x").expect("must match");
        assert_eq!(constant, c);
        assert_eq!(with_var, xk);
    }

    #[test]
    fn rewrite_ln_power_creates_product() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let x = ctx.var("x");
        let pow = ctx.add(Expr::Pow(a, x));
        let ln_pow = ctx.call_builtin(BuiltinFn::Ln, vec![pow]);
        let out = try_rewrite_ln_power(&mut ctx, ln_pow).expect("rewrite");
        assert!(matches!(ctx.get(out), Expr::Mul(_, _)));
    }

    #[test]
    fn build_detailed_collect_steps_falls_back_without_log_equation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let eq = Equation {
            lhs: x,
            rhs: x,
            op: cas_ast::RelOp::Eq,
        };

        let out = build_detailed_collect_steps(&mut ctx, None, &eq, "x");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].description, collect_and_factor_terms_message("x"));
    }

    #[test]
    fn build_detailed_collect_steps_emits_expand_step_for_distributive_shape() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let k = ctx.var("k");
        let m = ctx.var("m");
        let one_plus_x = ctx.add(Expr::Add(one, x));
        let lhs = ctx.add(Expr::Mul(k, one_plus_x));
        let rhs = ctx.add(Expr::Mul(x, m));
        let log_eq = Equation {
            lhs,
            rhs,
            op: cas_ast::RelOp::Eq,
        };

        let out = build_detailed_collect_steps(&mut ctx, Some(&log_eq), &log_eq, "x");
        assert!(!out.is_empty());
        assert_eq!(out[0].description, "Expand distributive law");
    }

    #[test]
    fn log_linear_marker_detection_matches_expected_step() {
        assert!(is_log_linear_take_log_step("Take log base e of both sides"));
        assert!(!is_log_linear_take_log_step("Square both sides"));
    }

    #[test]
    fn collect_and_factor_terms_message_formats_expected_text() {
        assert_eq!(
            collect_and_factor_terms_message("x"),
            "Collect and factor x terms"
        );
        assert_eq!(
            collect_and_factor_terms_message("t"),
            "Collect and factor t terms"
        );
    }
}
