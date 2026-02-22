use crate::isolation_utils::contains_var;
use cas_ast::{ordering::compare_expr, Constant, Context, Equation, Expr, ExprId};
use cas_math::build::mul2_raw;
use std::cmp::Ordering;

/// Build narration when a substitution variable is introduced.
pub fn detected_substitution_message(sub_expr_debug: &str) -> String {
    format!("Detected substitution: u = {}", sub_expr_debug)
}

/// Build narration for the equation rewritten in substitution variable.
pub fn substituted_equation_message(lhs_debug: &str, op_display: &str, rhs_debug: &str) -> String {
    format!(
        "Substituted equation: {} {} {}",
        lhs_debug, op_display, rhs_debug
    )
}

/// Build narration for back-substitution of a solved temporary variable.
pub fn back_substitute_message(lhs_debug: &str, rhs_debug: &str) -> String {
    format!("Back-substitute: {} = {}", lhs_debug, rhs_debug)
}

/// Didactic payload for one substitution step.
#[derive(Debug, Clone, PartialEq)]
pub struct SubstitutionDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// Didactic pair emitted when substitution is introduced (`u = ...` + rewritten equation).
#[derive(Debug, Clone, PartialEq)]
pub struct SubstitutionIntroDidacticSteps {
    pub detected: SubstitutionDidacticStep,
    pub rewritten: SubstitutionDidacticStep,
}

/// Build didactic payload for substitution detection (`u = expr`).
pub fn build_detected_substitution_step_with<F>(
    equation_after: Equation,
    sub_expr: ExprId,
    mut render_expr: F,
) -> SubstitutionDidacticStep
where
    F: FnMut(ExprId) -> String,
{
    let sub_expr_desc = render_expr(sub_expr);
    SubstitutionDidacticStep {
        description: detected_substitution_message(&sub_expr_desc),
        equation_after,
    }
}

/// Build didactic payload for rewritten equation in `u`.
pub fn build_substituted_equation_step_with<F>(
    equation_after: Equation,
    mut render_expr: F,
) -> SubstitutionDidacticStep
where
    F: FnMut(ExprId) -> String,
{
    let lhs_desc = render_expr(equation_after.lhs);
    let rhs_desc = render_expr(equation_after.rhs);
    let op_desc = equation_after.op.to_string();
    SubstitutionDidacticStep {
        description: substituted_equation_message(&lhs_desc, &op_desc, &rhs_desc),
        equation_after,
    }
}

/// Build didactic payload for back-substitution `expr = value`.
pub fn build_back_substitute_step_with<F>(
    equation_after: Equation,
    mut render_expr: F,
) -> SubstitutionDidacticStep
where
    F: FnMut(ExprId) -> String,
{
    let lhs_desc = render_expr(equation_after.lhs);
    let rhs_desc = render_expr(equation_after.rhs);
    SubstitutionDidacticStep {
        description: back_substitute_message(&lhs_desc, &rhs_desc),
        equation_after,
    }
}

/// Build didactic pair for substitution introduction:
/// 1) detected substitution `u = expr`
/// 2) equation rewritten in terms of `u`
pub fn build_substitution_intro_steps_with<F>(
    equation_before: Equation,
    substitution_expr: ExprId,
    rewritten_equation: Equation,
    mut render_expr: F,
) -> SubstitutionIntroDidacticSteps
where
    F: FnMut(ExprId) -> String,
{
    let detected =
        build_detected_substitution_step_with(equation_before, substitution_expr, &mut render_expr);
    let rewritten = build_substituted_equation_step_with(rewritten_equation, render_expr);
    SubstitutionIntroDidacticSteps {
        detected,
        rewritten,
    }
}

/// Exponential substitution rewrite extracted from strategy orchestration.
#[derive(Debug, Clone, PartialEq)]
pub struct ExponentialSubstitutionRewritePlan {
    pub substitution_expr: ExprId,
    pub equation: Equation,
}

/// Substitute a named variable with a value in an expression tree.
pub fn substitute_named_var(ctx: &mut Context, expr: ExprId, var: &str, value: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    match expr_data {
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var => value,
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => expr,

        Expr::Add(a, b) => {
            let a_sub = substitute_named_var(ctx, a, var, value);
            let b_sub = substitute_named_var(ctx, b, var, value);
            ctx.add(Expr::Add(a_sub, b_sub))
        }
        Expr::Sub(a, b) => {
            let a_sub = substitute_named_var(ctx, a, var, value);
            let b_sub = substitute_named_var(ctx, b, var, value);
            ctx.add(Expr::Sub(a_sub, b_sub))
        }
        Expr::Mul(a, b) => {
            let a_sub = substitute_named_var(ctx, a, var, value);
            let b_sub = substitute_named_var(ctx, b, var, value);
            ctx.add(Expr::Mul(a_sub, b_sub))
        }
        Expr::Div(a, b) => {
            let a_sub = substitute_named_var(ctx, a, var, value);
            let b_sub = substitute_named_var(ctx, b, var, value);
            ctx.add(Expr::Div(a_sub, b_sub))
        }
        Expr::Pow(a, b) => {
            let a_sub = substitute_named_var(ctx, a, var, value);
            let b_sub = substitute_named_var(ctx, b, var, value);
            ctx.add(Expr::Pow(a_sub, b_sub))
        }
        Expr::Neg(a) => {
            let a_sub = substitute_named_var(ctx, a, var, value);
            ctx.add(Expr::Neg(a_sub))
        }
        Expr::Function(name, args) => {
            let args_sub: Vec<_> = args
                .iter()
                .map(|&arg| substitute_named_var(ctx, arg, var, value))
                .collect();
            ctx.add(Expr::Function(name, args_sub))
        }
        Expr::Matrix { rows, cols, data } => {
            let data_sub: Vec<_> = data
                .iter()
                .map(|&elem| substitute_named_var(ctx, elem, var, value))
                .collect();
            ctx.add(Expr::Matrix {
                rows,
                cols,
                data: data_sub,
            })
        }
        Expr::Hold(inner) => {
            let inner_sub = substitute_named_var(ctx, inner, var, value);
            ctx.add(Expr::Hold(inner_sub))
        }
    }
}

fn is_e_base(ctx: &Context, base: ExprId) -> bool {
    match ctx.get(base) {
        Expr::Variable(sym_id) => ctx.sym_name(*sym_id) == "e",
        Expr::Constant(c) => matches!(c, Constant::E),
        _ => false,
    }
}

/// Returns true if `var` appears in `expr` outside of any `Pow(e, ...)` context.
fn var_outside_exponentials(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Variable(sym_id) => ctx.sym_name(*sym_id) == var,

        // Treat e^(...) as an opaque leaf: var inside exponent is allowed.
        Expr::Pow(b, _e) if is_e_base(ctx, *b) => false,

        Expr::Pow(b, e) => {
            var_outside_exponentials(ctx, *b, var) || var_outside_exponentials(ctx, *e, var)
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            var_outside_exponentials(ctx, *l, var) || var_outside_exponentials(ctx, *r, var)
        }
        Expr::Neg(e) | Expr::Hold(e) => var_outside_exponentials(ctx, *e, var),
        Expr::Function(_, args) => args.iter().any(|&a| var_outside_exponentials(ctx, a, var)),
        Expr::Matrix { data, .. } => data.iter().any(|&a| var_outside_exponentials(ctx, a, var)),
        Expr::SessionRef(_) | Expr::Number(_) | Expr::Constant(_) => false,
    }
}

fn collect_exponential_terms(ctx: &Context, expr: ExprId, var: &str, out: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Pow(b, e) => {
            if is_e_base(ctx, *b) && contains_var(ctx, *e, var) {
                out.push(expr);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            collect_exponential_terms(ctx, *l, var, out);
            collect_exponential_terms(ctx, *r, var, out);
        }
        Expr::Neg(e) | Expr::Hold(e) => collect_exponential_terms(ctx, *e, var, out),
        Expr::Function(_, args) => {
            for &a in args {
                collect_exponential_terms(ctx, a, var, out);
            }
        }
        Expr::Matrix { data, .. } => {
            for &a in data {
                collect_exponential_terms(ctx, a, var, out);
            }
        }
        Expr::SessionRef(_) | Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => {}
    }
}

/// Detect substitution candidate `u = e^x` (or base^x) for exponential equations.
///
/// Returns the base substitution term if the equation contains exponential
/// terms where `var` appears only in exponents.
pub fn detect_exponential_substitution(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<ExprId> {
    let mut terms = Vec::new();
    collect_exponential_terms(ctx, lhs, var, &mut terms);
    collect_exponential_terms(ctx, rhs, var, &mut terms);
    if terms.is_empty() {
        return None;
    }

    if var_outside_exponentials(ctx, lhs, var) || var_outside_exponentials(ctx, rhs, var) {
        return None;
    }

    let mut found_complex = false;
    let mut base_term = None;

    for term in &terms {
        if let Expr::Pow(_, exp) = ctx.get(*term) {
            if let Expr::Variable(sym_id) = ctx.get(*exp) {
                if ctx.sym_name(*sym_id) == var {
                    base_term = Some(*term);
                }
            } else if contains_var(ctx, *exp, var) {
                found_complex = true;
            }
        }
    }

    if !found_complex {
        return None;
    }

    if let Some(base) = base_term {
        return Some(base);
    }

    let inferred_base = match ctx.get(terms[0]) {
        Expr::Pow(base, _) => Some(*base),
        _ => None,
    };
    if let Some(base) = inferred_base {
        let var_expr = ctx.var(var);
        return Some(ctx.add(Expr::Pow(base, var_expr)));
    }

    None
}

/// Substitute `target` by `replacement` in an expression tree, with
/// special handling for exponential patterns: `e^(k*x) -> replacement^k`.
pub fn substitute_expr_pattern(
    ctx: &mut Context,
    expr: ExprId,
    target: ExprId,
    replacement: ExprId,
) -> ExprId {
    if compare_expr(ctx, expr, target) == Ordering::Equal {
        return replacement;
    }

    let expr_data = ctx.get(expr).clone();

    let target_pow = if let Expr::Pow(tb, te) = ctx.get(target) {
        Some((*tb, *te))
    } else {
        None
    };

    if let Expr::Pow(b, e) = &expr_data {
        if let Some((tb, te)) = target_pow {
            if compare_expr(ctx, *b, tb) == Ordering::Equal {
                let e_mul = if let Expr::Mul(l, r) = ctx.get(*e) {
                    Some((*l, *r))
                } else {
                    None
                };

                if let Some((l, r)) = e_mul {
                    let l_is_num = matches!(ctx.get(l), Expr::Number(_));
                    let r_is_num = matches!(ctx.get(r), Expr::Number(_));

                    let l_matches = compare_expr(ctx, l, te) == Ordering::Equal;
                    let r_matches = compare_expr(ctx, r, te) == Ordering::Equal;

                    if (l_matches && r_is_num) || (r_matches && l_is_num) {
                        let coeff = if l_matches { r } else { l };
                        return ctx.add(Expr::Pow(replacement, coeff));
                    }
                }
            }
        }
    }

    match expr_data {
        Expr::Add(l, r) => {
            let new_l = substitute_expr_pattern(ctx, l, target, replacement);
            let new_r = substitute_expr_pattern(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                return ctx.add(Expr::Add(new_l, new_r));
            }
        }
        Expr::Sub(l, r) => {
            let new_l = substitute_expr_pattern(ctx, l, target, replacement);
            let new_r = substitute_expr_pattern(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                return ctx.add(Expr::Sub(new_l, new_r));
            }
        }
        Expr::Mul(l, r) => {
            let new_l = substitute_expr_pattern(ctx, l, target, replacement);
            let new_r = substitute_expr_pattern(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                return mul2_raw(ctx, new_l, new_r);
            }
        }
        Expr::Div(l, r) => {
            let new_l = substitute_expr_pattern(ctx, l, target, replacement);
            let new_r = substitute_expr_pattern(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                return ctx.add(Expr::Div(new_l, new_r));
            }
        }
        Expr::Pow(b, e) => {
            let new_b = substitute_expr_pattern(ctx, b, target, replacement);
            let new_e = substitute_expr_pattern(ctx, e, target, replacement);
            if new_b != b || new_e != e {
                return ctx.add(Expr::Pow(new_b, new_e));
            }
        }
        Expr::Neg(e) => {
            let new_e = substitute_expr_pattern(ctx, e, target, replacement);
            if new_e != e {
                return ctx.add(Expr::Neg(new_e));
            }
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::new();
            let mut changed = false;
            for arg in args {
                let new_arg = substitute_expr_pattern(ctx, arg, target, replacement);
                if new_arg != arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                return ctx.add(Expr::Function(name, new_args));
            }
        }
        Expr::Matrix { rows, cols, data } => {
            let mut new_data = Vec::with_capacity(data.len());
            let mut changed = false;
            for elem in data {
                let new_elem = substitute_expr_pattern(ctx, elem, target, replacement);
                if new_elem != elem {
                    changed = true;
                }
                new_data.push(new_elem);
            }
            if changed {
                return ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                });
            }
        }
        Expr::Hold(inner) => {
            let new_inner = substitute_expr_pattern(ctx, inner, target, replacement);
            if new_inner != inner {
                return ctx.add(Expr::Hold(new_inner));
            }
        }
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
    }

    expr
}

/// Plan exponential substitution rewrite (`u = target`) for a solver equation.
///
/// Returns `None` when no safe substitution candidate is detected or when the
/// rewritten equation still contains the original variable.
pub fn plan_exponential_substitution_rewrite(
    ctx: &mut Context,
    equation: &Equation,
    var: &str,
    substitution_symbol: &str,
) -> Option<ExponentialSubstitutionRewritePlan> {
    let substitution_expr = detect_exponential_substitution(ctx, equation.lhs, equation.rhs, var)?;
    let substitution_var = ctx.var(substitution_symbol);
    let rewritten_lhs =
        substitute_expr_pattern(ctx, equation.lhs, substitution_expr, substitution_var);
    let rewritten_rhs =
        substitute_expr_pattern(ctx, equation.rhs, substitution_expr, substitution_var);

    if contains_var(ctx, rewritten_lhs, var) || contains_var(ctx, rewritten_rhs, var) {
        return None;
    }

    Some(ExponentialSubstitutionRewritePlan {
        substitution_expr,
        equation: Equation {
            lhs: rewritten_lhs,
            rhs: rewritten_rhs,
            op: equation.op.clone(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitute_simple_named_var() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let expr = ctx.add(Expr::Add(x, one));

        let replaced = substitute_named_var(&mut ctx, expr, "x", y);
        match ctx.get(replaced) {
            Expr::Add(lhs, rhs) => {
                let lhs_is_y =
                    matches!(ctx.get(*lhs), Expr::Variable(sym) if ctx.sym_name(*sym) == "y");
                let rhs_is_y =
                    matches!(ctx.get(*rhs), Expr::Variable(sym) if ctx.sym_name(*sym) == "y");
                let lhs_is_one = matches!(
                    ctx.get(*lhs),
                    Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
                );
                let rhs_is_one = matches!(
                    ctx.get(*rhs),
                    Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
                );
                assert!(
                    (lhs_is_y && rhs_is_one) || (rhs_is_y && lhs_is_one),
                    "expected Add(y, 1) in canonical order, got lhs={:?}, rhs={:?}",
                    ctx.get(*lhs),
                    ctx.get(*rhs)
                );
            }
            other => panic!("expected Add after substitution, got {other:?}"),
        }
    }

    #[test]
    fn detect_exponential_substitution_finds_e_pow_x_base() {
        let mut ctx = Context::new();
        let e = ctx.add(Expr::Constant(Constant::E));
        let x = ctx.var("x");
        let two = ctx.num(2);
        let e_pow_x = ctx.add(Expr::Pow(e, x));
        let two_x = ctx.add(Expr::Mul(two, x));
        let e_pow_2x = ctx.add(Expr::Pow(e, two_x));
        let sum = ctx.add(Expr::Add(e_pow_2x, e_pow_x));
        let one = ctx.num(1);
        let sub = detect_exponential_substitution(&mut ctx, sum, one, "x")
            .expect("must detect substitution base");
        assert_eq!(sub, e_pow_x);
    }

    #[test]
    fn substitute_expr_pattern_handles_e_pow_2x() {
        let mut ctx = Context::new();
        let e = ctx.add(Expr::Constant(Constant::E));
        let x = ctx.var("x");
        let two = ctx.num(2);
        let u = ctx.var("u");
        let e_pow_x = ctx.add(Expr::Pow(e, x));
        let two_x = ctx.add(Expr::Mul(two, x));
        let e_pow_2x = ctx.add(Expr::Pow(e, two_x));
        let out = substitute_expr_pattern(&mut ctx, e_pow_2x, e_pow_x, u);
        match ctx.get(out) {
            Expr::Pow(base, exp) => {
                assert_eq!(*base, u);
                assert_eq!(*exp, two);
            }
            other => panic!("expected Pow(u,2), got {other:?}"),
        }
    }

    #[test]
    fn plan_exponential_substitution_rewrite_builds_safe_u_equation() {
        let mut ctx = Context::new();
        let e = ctx.add(Expr::Constant(Constant::E));
        let x = ctx.var("x");
        let two = ctx.num(2);
        let one = ctx.num(1);
        let e_pow_x = ctx.add(Expr::Pow(e, x));
        let two_x = ctx.add(Expr::Mul(two, x));
        let e_pow_2x = ctx.add(Expr::Pow(e, two_x));
        let lhs = ctx.add(Expr::Add(e_pow_2x, e_pow_x));
        let eq = Equation {
            lhs,
            rhs: one,
            op: cas_ast::RelOp::Eq,
        };

        let plan = plan_exponential_substitution_rewrite(&mut ctx, &eq, "x", "u")
            .expect("should build substitution rewrite");
        assert_eq!(plan.substitution_expr, e_pow_x);
        assert!(!contains_var(&ctx, plan.equation.lhs, "x"));
        assert!(!contains_var(&ctx, plan.equation.rhs, "x"));
    }

    #[test]
    fn plan_exponential_substitution_rewrite_rejects_mixed_variable_positions() {
        let mut ctx = Context::new();
        let e = ctx.add(Expr::Constant(Constant::E));
        let x = ctx.var("x");
        let one = ctx.num(1);
        let e_pow_x = ctx.add(Expr::Pow(e, x));
        let lhs = ctx.add(Expr::Add(x, e_pow_x));
        let eq = Equation {
            lhs,
            rhs: one,
            op: cas_ast::RelOp::Eq,
        };

        assert!(plan_exponential_substitution_rewrite(&mut ctx, &eq, "x", "u").is_none());
    }

    #[test]
    fn substitution_didactic_messages_format_expected_text() {
        assert_eq!(
            detected_substitution_message("ExprId(42)"),
            "Detected substitution: u = ExprId(42)"
        );
        assert_eq!(
            substituted_equation_message("ExprId(1)", "=", "ExprId(2)"),
            "Substituted equation: ExprId(1) = ExprId(2)"
        );
        assert_eq!(
            back_substitute_message("ExprId(3)", "ExprId(4)"),
            "Back-substitute: ExprId(3) = ExprId(4)"
        );
    }

    #[test]
    fn substitution_step_builders_use_rendered_payloads() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: cas_ast::RelOp::Eq,
        };

        let detect = build_detected_substitution_step_with(eq.clone(), x, |_| "exp".to_string());
        assert_eq!(detect.description, "Detected substitution: u = exp");

        let rewritten = build_substituted_equation_step_with(eq.clone(), |_| "u".to_string());
        assert_eq!(rewritten.description, "Substituted equation: u = u");

        let back = build_back_substitute_step_with(eq.clone(), |_| "v".to_string());
        assert_eq!(back.description, "Back-substitute: v = v");
    }

    #[test]
    fn build_substitution_intro_steps_with_builds_detected_and_rewritten_steps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq_before = Equation {
            lhs: x,
            rhs: y,
            op: cas_ast::RelOp::Eq,
        };
        let eq_after = Equation {
            lhs: y,
            rhs: x,
            op: cas_ast::RelOp::Eq,
        };

        let steps =
            build_substitution_intro_steps_with(eq_before.clone(), x, eq_after.clone(), |_| {
                "u".to_string()
            });

        assert_eq!(steps.detected.description, "Detected substitution: u = u");
        assert_eq!(steps.detected.equation_after, eq_before);
        assert_eq!(steps.rewritten.description, "Substituted equation: u = u");
        assert_eq!(steps.rewritten.equation_after, eq_after);
    }
}
