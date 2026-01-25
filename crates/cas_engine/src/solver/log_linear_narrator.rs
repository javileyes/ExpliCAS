//! Log-linear equation narrator for didactic step display.
//!
//! This module rewrites solve steps for log-linear equations (like `3^(x+1) = 5^x`)
//! into a coherent, atomic step sequence that follows a natural pedagogical flow:
//!
//! 1. Take log of both sides
//! 2. Apply power rule: ln(a^x) → x·ln(a)
//! 3. Expand products: ln(3)·(1+x) → ln(3) + x·ln(3)
//! 4. Move x terms to one side
//! 5. Factor out x
//! 6. Apply log quotient rule: ln(a) - ln(b) → ln(a/b)
//! 7. Divide
//!
//! The key insight is that the MATHEMATICAL result is correct, but the TRACE
//! needs to be restructured for didactic clarity.

use crate::helpers::is_one;
use crate::solver::SolveStep;
use cas_ast::{BuiltinFn, Context, Equation, Expr, ExprId};

// =============================================================================
// Display-only cleanup: remove "1·" identity coefficients from equations
// =============================================================================

/// Strip "1*expr" patterns from an expression for cleaner display.
/// This is a display-only transformation - it doesn't affect the mathematical result.
///
/// Examples:
/// - `1 * ln(3)` → `ln(3)`
/// - `ln(3) * 1` → `ln(3)`
/// - `1 * x * ln(5)` → `x * ln(5)`
fn strip_mul_one(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    match expr_data {
        Expr::Mul(l, r) => {
            // First recursively clean both sides
            let clean_l = strip_mul_one(ctx, l);
            let clean_r = strip_mul_one(ctx, r);

            // Then check for 1*x or x*1
            if is_one(ctx, clean_l) {
                return clean_r;
            }
            if is_one(ctx, clean_r) {
                return clean_l;
            }

            // Neither is 1, keep the multiplication
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
        // Other expressions pass through unchanged
        _ => expr,
    }
}

/// Apply strip_mul_one to both sides of an equation.
fn strip_equation(ctx: &mut Context, eq: &Equation) -> Equation {
    Equation {
        lhs: strip_mul_one(ctx, eq.lhs),
        rhs: strip_mul_one(ctx, eq.rhs),
        op: eq.op.clone(),
    }
}

/// Check if a step sequence is a log-linear solve pattern.
/// Returns true if we should rewrite the steps for better didactic display.
pub fn is_log_linear_pattern(steps: &[SolveStep]) -> bool {
    if steps.is_empty() {
        return false;
    }

    // Pattern: First step is "Take log base e of both sides"
    steps[0].description == "Take log base e of both sides"
}

/// Rewrite log-linear solve steps into a coherent didactic sequence.
///
/// This analyzes the current steps and the equation transformations to
/// produce a cleaner narrative. It does NOT change the mathematical result,
/// only how the steps are presented.
///
/// # Arguments
/// * `ctx` - Expression context
/// * `steps` - Original solve steps  
/// * `detailed` - If true, decompose "Collect and factor" into atomic sub-steps
pub fn rewrite_log_linear_steps(
    ctx: &mut Context,
    steps: Vec<SolveStep>,
    detailed: bool,
) -> Vec<SolveStep> {
    if steps.is_empty() || !is_log_linear_pattern(&steps) {
        return steps;
    }

    // Find the key steps in the original sequence
    let mut result = Vec::new();
    let mut i = 0;

    // Track the equation after "Take log" for building intermediates
    let mut log_eq: Option<Equation> = None;

    while i < steps.len() {
        let step = &steps[i];

        // Step 1: Keep "Take log base e" but ensure RHS shows x·ln(b) not ln(b^x)
        if step.description == "Take log base e of both sides" {
            // Check if RHS is ln(something^x) and rewrite to x·ln(something)
            let improved_rhs = try_rewrite_log_power(ctx, step.equation_after.rhs);

            let eq = Equation {
                lhs: step.equation_after.lhs,
                rhs: improved_rhs.unwrap_or(step.equation_after.rhs),
                op: step.equation_after.op.clone(),
            };

            // Save for building intermediates
            log_eq = Some(eq.clone());

            result.push(SolveStep {
                description: "Take log base e of both sides".to_string(),
                equation_after: eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });

            i += 1;
            continue;
        }

        // Handle "Collect terms in x" - decompose if detailed mode
        if step.description.starts_with("Collect terms in") {
            if detailed {
                // Decompose into atomic sub-steps with REAL intermediate equations
                let sub_steps =
                    generate_detailed_collect_steps_v2(ctx, log_eq.as_ref(), &step.equation_after);
                result.extend(sub_steps);
            } else {
                // Compact mode: single step with cleaner description
                result.push(SolveStep {
                    description: "Collect and factor x terms".to_string(),
                    equation_after: step.equation_after.clone(),
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            i += 1;
            continue;
        }

        // Default: keep the step
        result.push(step.clone());
        i += 1;
    }

    result
}

/// Generate detailed sub-steps with REAL intermediate equations.
///
/// Given the equation after "Take log" (e.g., `ln(3)·(1+x) = x·ln(5)`)
/// and the final equation (e.g., `ln(3) + x·ln(3/5) = 0`),
/// construct intermediate states:
///
/// 1. Expand: `ln(3) + x·ln(3) = x·ln(5)`
/// 2. Move x: `ln(3) = x·ln(5) - x·ln(3)`
/// 3. Factor: `ln(3) = x·(ln(5) - ln(3))`
/// 4. Log quotient (if applicable): `ln(3) = x·ln(5/3)`
fn generate_detailed_collect_steps_v2(
    ctx: &mut Context,
    log_eq_opt: Option<&Equation>,
    final_eq: &Equation,
) -> Vec<SolveStep> {
    let mut sub_steps = Vec::new();

    // If we don't have the log equation, fall back to single step
    let log_eq = match log_eq_opt {
        Some(eq) => eq,
        None => {
            sub_steps.push(SolveStep {
                description: "Collect and factor x terms".to_string(),
                equation_after: final_eq.clone(),
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
            return sub_steps;
        }
    };

    // Try to detect pattern: Mul(K, Add(1, x)) = Mul(x, Ln(b))
    // where K is ln(a) for some constant a

    // Step 1: Expand distributive law
    // If LHS is K*(1+x), expand to K + K*x
    if let Some(expanded_lhs) = try_expand_distributive(ctx, log_eq.lhs) {
        let expand_eq = Equation {
            lhs: expanded_lhs,
            rhs: log_eq.rhs,
            op: cas_ast::RelOp::Eq,
        };
        let clean_expand_eq = strip_equation(ctx, &expand_eq);
        sub_steps.push(SolveStep {
            description: "Expand distributive law".to_string(),
            equation_after: clean_expand_eq,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });

        // Step 2: Move x terms to one side
        // From: K + K*x = x*M  ->  K = x*M - K*x
        if let Some((constant, x_coef_lhs)) = try_extract_constant_and_x_term(ctx, expanded_lhs) {
            let x_coef_rhs = log_eq.rhs; // This is x*ln(5) or similar

            // Build: K = x*M - x*K (actually: K = RHS - x_coef_lhs)
            let moved_rhs = ctx.add(Expr::Sub(x_coef_rhs, x_coef_lhs));
            let move_eq = Equation {
                lhs: constant,
                rhs: moved_rhs,
                op: cas_ast::RelOp::Eq,
            };
            let clean_move_eq = strip_equation(ctx, &move_eq);
            sub_steps.push(SolveStep {
                description: "Move x terms to one side".to_string(),
                equation_after: clean_move_eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });

            // Step 3: Factor out x
            // From: K = x*M - x*K' -> K = x*(M - K')
            // Extract x from both terms and combine coefficients
            if let Some(factored_rhs) = try_factor_x(ctx, moved_rhs) {
                let factor_eq = Equation {
                    lhs: constant,
                    rhs: factored_rhs,
                    op: cas_ast::RelOp::Eq,
                };
                let clean_factor_eq = strip_equation(ctx, &factor_eq);
                sub_steps.push(SolveStep {
                    description: "Factor out x".to_string(),
                    equation_after: clean_factor_eq,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
        }
    }

    // If we couldn't decompose, add the final step
    if sub_steps.is_empty() {
        sub_steps.push(SolveStep {
            description: "Collect and factor x terms".to_string(),
            equation_after: final_eq.clone(),
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    sub_steps
}

/// Try to expand K*(A+B) into K*A + K*B
fn try_expand_distributive(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Mul(l, r) = ctx.get(expr).clone() {
        // Check if one side is Add
        if let Expr::Add(a, b) = ctx.get(r).clone() {
            // l * (a + b) -> l*a + l*b
            let term1 = ctx.add(Expr::Mul(l, a));
            let term2 = ctx.add(Expr::Mul(l, b));
            return Some(ctx.add(Expr::Add(term1, term2)));
        }
        if let Expr::Add(a, b) = ctx.get(l).clone() {
            // (a + b) * r -> a*r + b*r
            let term1 = ctx.add(Expr::Mul(a, r));
            let term2 = ctx.add(Expr::Mul(b, r));
            return Some(ctx.add(Expr::Add(term1, term2)));
        }
    }
    None
}

/// Extract (constant, x_term) from an expression like K + x*M
fn try_extract_constant_and_x_term(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if let Expr::Add(l, r) = ctx.get(expr) {
        // Check which side contains x
        let l_has_x = expr_contains_var(ctx, *l, "x");
        let r_has_x = expr_contains_var(ctx, *r, "x");

        match (l_has_x, r_has_x) {
            (false, true) => Some((*l, *r)), // l is constant, r has x
            (true, false) => Some((*r, *l)), // r is constant, l has x
            _ => None,
        }
    } else {
        None
    }
}

/// Try to factor x from an expression like x*A - x*B into x*(A-B)
fn try_factor_x(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Sub(l, r) => {
            // Extract x coefficient from both terms
            let l_coef = try_extract_x_coefficient(ctx, l)?;
            let r_coef = try_extract_x_coefficient(ctx, r)?;

            // Build x * (l_coef - r_coef)
            let coef_diff = ctx.add(Expr::Sub(l_coef, r_coef));
            let x = ctx.var("x");
            Some(ctx.add(Expr::Mul(x, coef_diff)))
        }
        Expr::Add(l, r) => {
            let l_coef = try_extract_x_coefficient(ctx, l)?;
            let r_coef = try_extract_x_coefficient(ctx, r)?;

            let coef_sum = ctx.add(Expr::Add(l_coef, r_coef));
            let x = ctx.var("x");
            Some(ctx.add(Expr::Mul(x, coef_sum)))
        }
        _ => None,
    }
}

/// Extract coefficient of x from term like x*K
fn try_extract_x_coefficient(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Check if l or r is just "x"
        if matches!(ctx.get(*l), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == "x") {
            return Some(*r);
        }
        if matches!(ctx.get(*r), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == "x") {
            return Some(*l);
        }
    }
    // Check if expr is just "x" (coef = 1)
    if matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == "x") {
        // Would need to return 1, but we can't easily do this here
        return None;
    }
    None
}

/// Check if expression contains a specific variable
fn expr_contains_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    use crate::solver::isolation::contains_var;
    contains_var(ctx, expr, var)
}

/// Try to rewrite ln(a^x) → x·ln(a) for cleaner display.
/// Returns Some(new_expr) if rewrite was done, None otherwise.
fn try_rewrite_log_power(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let expr_data = ctx.get(expr).clone();

    match expr_data {
        // ln(a^exp) → exp·ln(a)
        Expr::Function(fn_id, args) if ctx.is_builtin(fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            let inner = args[0];
            if let Expr::Pow(base, exp) = ctx.get(inner).clone() {
                // Create exp * ln(base)
                let ln_base = ctx.call("ln", vec![base]);
                let product = ctx.add(Expr::Mul(exp, ln_base));
                return Some(product);
            }
            None
        }
        _ => None,
    }
}

/// Check if equation after log contains pattern ln(a)·(1+x) = x·ln(b)
/// This indicates a log-linear equation we can narrate better.
pub fn detect_log_linear_form(ctx: &Context, lhs: ExprId, rhs: ExprId, var: &str) -> bool {
    // Check if LHS has pattern ln(a)·f(x) and RHS has pattern g(x)·ln(b)
    let lhs_has_ln = contains_ln_times_var(ctx, lhs, var);
    let rhs_has_ln = contains_ln_times_var(ctx, rhs, var);

    lhs_has_ln && rhs_has_ln
}

/// Check if expression contains a pattern like ln(c) * (something with var)
fn contains_ln_times_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            // Check if one side is ln(...) and other contains var
            let l_is_ln = is_ln_call(ctx, *l);
            let r_is_ln = is_ln_call(ctx, *r);
            let l_has_var = crate::solver::contains_var(ctx, *l, var);
            let r_has_var = crate::solver::contains_var(ctx, *r, var);

            (l_is_ln && r_has_var) || (r_is_ln && l_has_var) || (l_has_var && r_has_var)
        }
        Expr::Add(l, r) | Expr::Sub(l, r) => {
            contains_ln_times_var(ctx, *l, var) || contains_ln_times_var(ctx, *r, var)
        }
        _ => false,
    }
}

/// Check if expression is a ln(...) call
fn is_ln_call(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Function(name, _) if ctx.is_builtin(*name, BuiltinFn::Ln))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_log_linear_pattern() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let steps = vec![SolveStep {
            description: "Take log base e of both sides".to_string(),
            equation_after: Equation {
                lhs: x,
                rhs: x,
                op: cas_ast::RelOp::Eq,
            },
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        }];

        assert!(is_log_linear_pattern(&steps));
    }

    #[test]
    fn test_not_log_linear_pattern() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let steps = vec![SolveStep {
            description: "Square both sides".to_string(),
            equation_after: Equation {
                lhs: x,
                rhs: x,
                op: cas_ast::RelOp::Eq,
            },
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        }];

        assert!(!is_log_linear_pattern(&steps));
    }
}
