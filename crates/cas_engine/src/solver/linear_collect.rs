//! Linear Collect Strategy for solving equations with additive terms.
//!
//! This module handles equations where the target variable appears in multiple
//! additive terms, like A = P + P*r*t. It factors out the variable and solves
//! by division, returning a Conditional solution when the coefficient might be zero.
//!
//! Example: A = P + P*r*t
//! 1. Move all to LHS: P + P*r*t - A = 0
//! 2. Factor: P*(1 + r*t) - A = 0
//! 3. Solve: P = A / (1 + r*t)  [guard: 1 + r*t ≠ 0]

use cas_ast::{Case, ConditionPredicate, ConditionSet, Context, Expr, ExprId, RelOp, SolutionSet};

use crate::engine::Simplifier;
use crate::nary::{add_terms_signed, Sign};
use crate::solver::isolation::contains_var;
use crate::solver::SolveStep;

/// Classification of a term with respect to a variable.
#[derive(Debug)]
pub enum TermClass {
    /// Term doesn't contain the variable (e.g., A, 5, r*t)
    Const(ExprId),
    /// Term is linear in the variable: coef * var (e.g., P, 3*P, r*t*P)
    /// None means coefficient is 1 (implicit)
    Linear(Option<ExprId>), // the coefficient (without var), None = 1
    /// Term contains variable in non-linear way (e.g., P^2, sqrt(P), 1/P)
    NonLinear,
}

/// Try to solve a linear equation where variable appears in multiple additive terms.
///
/// Returns Some((SolutionSet, steps)) if successful, None if not applicable.
///
/// Example: P + P*r*t - A = 0 → P = A / (1 + r*t) with guard 1+r*t ≠ 0
pub(crate) fn try_linear_collect(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<(SolutionSet, Vec<SolveStep>)> {
    let ctx = &mut simplifier.context;

    // 1. Build expr = lhs - rhs (move everything to LHS, so expr = 0)
    let expr = ctx.add(Expr::Sub(lhs, rhs));
    let (expr, _) = simplifier.simplify(expr);

    // 2. Flatten as sum of SIGNED terms using canonical utility
    let terms = add_terms_signed(&simplifier.context, expr);

    // 3. Classify each term, respecting signs
    let mut coeff_parts: Vec<ExprId> = Vec::new();
    let mut const_parts: Vec<ExprId> = Vec::new();

    for (term, sign) in terms {
        match split_linear_term(&mut simplifier.context, term, var) {
            TermClass::Const(_) => {
                // Apply sign to constant term
                let signed_term = match sign {
                    Sign::Pos => term,
                    Sign::Neg => simplifier.context.add(Expr::Neg(term)),
                };
                const_parts.push(signed_term);
            }
            TermClass::Linear(c) => {
                // Convert None (implicit 1) to explicit 1
                let coef = c.unwrap_or_else(|| simplifier.context.num(1));
                // Apply sign to coefficient
                let signed_coef = match sign {
                    Sign::Pos => coef,
                    Sign::Neg => simplifier.context.add(Expr::Neg(coef)),
                };
                coeff_parts.push(signed_coef);
            }
            TermClass::NonLinear => {
                // Variable appears non-linearly, this strategy doesn't apply
                return None;
            }
        }
    }

    // If no linear terms found, strategy doesn't apply
    if coeff_parts.is_empty() {
        return None;
    }

    // 4. Build coeff = sum of linear coefficients
    let coeff = build_sum(&mut simplifier.context, &coeff_parts);
    let (coeff, _) = simplifier.simplify(coeff);

    // 5. Build const = sum of constant parts (with sign flipped for solution)
    // coeff*var + const = 0 → var = -const / coeff
    let const_sum = build_sum(&mut simplifier.context, &const_parts);
    let neg_const = simplifier.context.add(Expr::Neg(const_sum));
    let (neg_const, _) = simplifier.simplify(neg_const);

    // 6. Build solution: var = -const / coeff
    let solution = simplifier.context.add(Expr::Div(neg_const, coeff));
    let (solution, _) = simplifier.simplify(solution);

    // 7. Build step description
    let mut steps = Vec::new();
    if simplifier.collect_steps() {
        let var_expr = simplifier.context.var(var);
        steps.push(SolveStep {
            description: format!(
                "Collect terms in {} and factor: {} · {} = {}",
                var,
                cas_ast::DisplayExpr {
                    context: &simplifier.context,
                    id: coeff
                },
                var,
                cas_ast::DisplayExpr {
                    context: &simplifier.context,
                    id: neg_const
                }
            ),
            equation_after: cas_ast::Equation {
                lhs: simplifier.context.add(Expr::Mul(coeff, var_expr)),
                rhs: neg_const,
                op: RelOp::Eq,
            },
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
        steps.push(SolveStep {
            description: format!(
                "Divide both sides by {}",
                cas_ast::DisplayExpr {
                    context: &simplifier.context,
                    id: coeff
                }
            ),
            equation_after: cas_ast::Equation {
                lhs: var_expr,
                rhs: solution,
                op: RelOp::Eq,
            },
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    // 8. Check if we can give a direct solution (coef is a ground constant)
    // If coef contains no variables and prove_nonzero returns Proven, skip the Conditional
    if !contains_var(&simplifier.context, coeff, var) {
        use crate::domain::Proof;
        use crate::helpers::prove_nonzero;

        let proof = prove_nonzero(&simplifier.context, coeff);
        if proof == Proof::Proven {
            // Direct solution: coeff is provably non-zero constant
            return Some((SolutionSet::Discrete(vec![solution]), steps));
        }
        // If Disproven (coeff == 0), check neg_const
        if proof == Proof::Disproven {
            let const_proof = prove_nonzero(&simplifier.context, neg_const);
            if const_proof == Proof::Disproven {
                // Both are 0: 0*x + 0 = 0 → AllReals
                return Some((SolutionSet::AllReals, steps));
            } else if const_proof == Proof::Proven {
                // 0*x + nonzero = 0 → Empty
                return Some((SolutionSet::Empty, steps));
            }
            // Otherwise fall through to Conditional
        }
    }

    // 9. Return as Conditional with guard: coeff ≠ 0
    // Primary case: coeff ≠ 0 → { solution }
    let guard = ConditionSet::single(ConditionPredicate::NonZero(coeff));
    let primary_case = Case::new(guard, SolutionSet::Discrete(vec![solution]));

    // Degenerate case: coeff = 0
    // If coeff = 0 AND const = 0 → AllReals
    // If coeff = 0 AND const ≠ 0 → EmptySet

    // Case: coeff = 0 ∧ const = 0 → AllReals
    // Note: we use neg_const (= -const_sum = A for our example) for cleaner display
    let mut both_zero_guard = ConditionSet::single(ConditionPredicate::EqZero(coeff));
    both_zero_guard.push(ConditionPredicate::EqZero(neg_const));
    let all_reals_case = Case::new(both_zero_guard, SolutionSet::AllReals);

    // Case: otherwise → EmptySet
    let otherwise_case = Case::new(
        ConditionSet::empty(), // "otherwise"
        SolutionSet::Empty,
    );

    let conditional = SolutionSet::Conditional(vec![primary_case, all_reals_case, otherwise_case]);

    Some((conditional, steps))
}

/// Classify a term as Const, Linear(coef), or NonLinear.
fn split_linear_term(ctx: &mut Context, term: ExprId, var: &str) -> TermClass {
    // If term doesn't contain var, it's a constant
    if !contains_var(ctx, term, var) {
        return TermClass::Const(term);
    }

    match ctx.get(term).clone() {
        // term == var → Linear(1)
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var => {
            // Coefficient is 1 (implicit)
            TermClass::Linear(None) // None = implicit coefficient 1
        }
        // term == coef * var or var * coef → Linear(coef)
        Expr::Mul(l, r) => {
            let l_has = contains_var(ctx, l, var);
            let r_has = contains_var(ctx, r, var);

            match (l_has, r_has) {
                (true, false) => {
                    // l contains var, r is coefficient
                    // Check if l is just the variable
                    if matches!(ctx.get(l), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var)
                    {
                        TermClass::Linear(Some(r))
                    } else {
                        // l is more complex (Mul(P, k), etc.)
                        // Recursively check l
                        match split_linear_term(ctx, l, var) {
                            TermClass::Linear(inner_coef) => {
                                // Combine: r * inner_coef
                                match inner_coef {
                                    Some(k) => {
                                        let combined = ctx.add(Expr::Mul(r, k));
                                        TermClass::Linear(Some(combined))
                                    }
                                    None => {
                                        // inner was just var, so coef = r
                                        TermClass::Linear(Some(r))
                                    }
                                }
                            }
                            _ => TermClass::NonLinear,
                        }
                    }
                }
                (false, true) => {
                    // r contains var, l is coefficient
                    if matches!(ctx.get(r), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var)
                    {
                        TermClass::Linear(Some(l))
                    } else {
                        match split_linear_term(ctx, r, var) {
                            TermClass::Linear(inner_coef) => {
                                // Combine: l * inner_coef
                                match inner_coef {
                                    Some(k) => {
                                        let combined = ctx.add(Expr::Mul(l, k));
                                        TermClass::Linear(Some(combined))
                                    }
                                    None => TermClass::Linear(Some(l)),
                                }
                            }
                            _ => TermClass::NonLinear,
                        }
                    }
                }
                (true, true) => {
                    // Both contain var (e.g., var * var) → NonLinear
                    TermClass::NonLinear
                }
                (false, false) => {
                    // Neither contains var - shouldn't happen since we checked above
                    TermClass::Const(term)
                }
            }
        }
        // term == -expr → check inner
        Expr::Neg(inner) => match split_linear_term(ctx, inner, var) {
            TermClass::Const(_) => TermClass::Const(term), // Keep negated form
            TermClass::Linear(_) => TermClass::Linear(Some(term)), // Negate term = negate coef
            TermClass::NonLinear => TermClass::NonLinear,
        },
        // var in power position → NonLinear (unless power is 1)
        Expr::Pow(base, exp) => {
            if contains_var(ctx, exp, var) {
                // Variable in exponent → NonLinear
                TermClass::NonLinear
            } else if contains_var(ctx, base, var) {
                // Check if exponent is 1
                if let Expr::Number(n) = ctx.get(exp) {
                    if *n == num_rational::BigRational::from_integer(1.into()) {
                        // base^1 = base
                        return split_linear_term(ctx, base, var);
                    }
                }
                // base^n with n ≠ 1 → NonLinear
                TermClass::NonLinear
            } else {
                TermClass::Const(term)
            }
        }
        // Division: if var in denominator → NonLinear
        Expr::Div(num, denom) => {
            if contains_var(ctx, denom, var) {
                TermClass::NonLinear
            } else {
                // var only in numerator: num = coef * var, result = (coef/denom) * var
                match split_linear_term(ctx, num, var) {
                    TermClass::Linear(_) => TermClass::NonLinear, // Too complex for now
                    TermClass::Const(_) => TermClass::Const(term),
                    TermClass::NonLinear => TermClass::NonLinear,
                }
            }
        }
        // Functions: if var inside → NonLinear
        Expr::Function(_, args) => {
            if args.iter().any(|a| contains_var(ctx, *a, var)) {
                TermClass::NonLinear
            } else {
                TermClass::Const(term)
            }
        }
        // Anything else with var → NonLinear
        _ => TermClass::NonLinear,
    }
}

/// Build a sum expression from a list of terms.
fn build_sum(ctx: &mut Context, parts: &[ExprId]) -> ExprId {
    if parts.is_empty() {
        return ctx.num(0);
    }
    let mut result = parts[0];
    for &part in &parts[1..] {
        result = ctx.add(Expr::Add(result, part));
    }
    result
}

// ============================================================================
// NEW: Structural Linear Form Extraction
// ============================================================================

/// Represents the linear form of an expression: coef * var + constant
/// Both coef and constant are var-free expressions.
#[derive(Debug, Clone)]
pub struct LinearForm {
    /// Coefficient of the variable (var-free)
    pub coef: ExprId,
    /// Constant term (var-free)
    pub constant: ExprId,
}

/// Extract the linear form of an expression with respect to a variable.
///
/// Returns `Some((coef, constant))` where `expr = coef * var + constant`,
/// with both `coef` and `constant` being var-free expressions.
///
/// Returns `None` if the expression is non-linear in the variable
/// (e.g., var^2, sin(var), var in denominator).
///
/// # Examples
/// - `linear_form(x, "x")` → `(1, 0)`
/// - `linear_form(y, "x")` → `(0, y)`
/// - `linear_form(x + 1, "x")` → `(1, 1)`
/// - `linear_form(y*(x+1), "x")` → `(y, y)` (because y*x + y*1)
/// - `linear_form(x^2, "x")` → `None`
pub(crate) fn linear_form(ctx: &mut Context, expr: ExprId, var: &str) -> Option<LinearForm> {
    // Base case: doesn't contain var → constant
    if !contains_var(ctx, expr, var) {
        let zero = ctx.num(0);
        return Some(LinearForm {
            coef: zero,
            constant: expr,
        });
    }

    match ctx.get(expr).clone() {
        // var itself → (1, 0)
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var => {
            let one = ctx.num(1);
            let zero = ctx.num(0);
            Some(LinearForm {
                coef: one,
                constant: zero,
            })
        }

        // Other variable (shouldn't reach here due to contains_var check above)
        Expr::Variable(_) => {
            let zero = ctx.num(0);
            Some(LinearForm {
                coef: zero,
                constant: expr,
            })
        }

        // Add(u, v) → (a1 + a2, b1 + b2)
        Expr::Add(u, v) => {
            let lf_u = linear_form(ctx, u, var)?;
            let lf_v = linear_form(ctx, v, var)?;
            let coef = ctx.add(Expr::Add(lf_u.coef, lf_v.coef));
            let constant = ctx.add(Expr::Add(lf_u.constant, lf_v.constant));
            Some(LinearForm { coef, constant })
        }

        // Sub(u, v) → (a1 - a2, b1 - b2)
        Expr::Sub(u, v) => {
            let lf_u = linear_form(ctx, u, var)?;
            let lf_v = linear_form(ctx, v, var)?;
            let coef = ctx.add(Expr::Sub(lf_u.coef, lf_v.coef));
            let constant = ctx.add(Expr::Sub(lf_u.constant, lf_v.constant));
            Some(LinearForm { coef, constant })
        }

        // Neg(u) → (-a, -b)
        Expr::Neg(u) => {
            let lf_u = linear_form(ctx, u, var)?;
            let coef = ctx.add(Expr::Neg(lf_u.coef));
            let constant = ctx.add(Expr::Neg(lf_u.constant));
            Some(LinearForm { coef, constant })
        }

        // Mul(u, v)
        Expr::Mul(u, v) => {
            let u_has = contains_var(ctx, u, var);
            let v_has = contains_var(ctx, v, var);

            match (u_has, v_has) {
                // Both contain var → non-linear (e.g., x * x)
                (true, true) => None,

                // u var-free, v contains var: u * (a*x + b) = (u*a)*x + (u*b)
                (false, true) => {
                    let lf_v = linear_form(ctx, v, var)?;
                    let coef = ctx.add(Expr::Mul(u, lf_v.coef));
                    let constant = ctx.add(Expr::Mul(u, lf_v.constant));
                    Some(LinearForm { coef, constant })
                }

                // u contains var, v var-free: (a*x + b) * v = (a*v)*x + (b*v)
                (true, false) => {
                    let lf_u = linear_form(ctx, u, var)?;
                    let coef = ctx.add(Expr::Mul(lf_u.coef, v));
                    let constant = ctx.add(Expr::Mul(lf_u.constant, v));
                    Some(LinearForm { coef, constant })
                }

                // Neither contains var (shouldn't reach here)
                (false, false) => {
                    let zero = ctx.num(0);
                    Some(LinearForm {
                        coef: zero,
                        constant: expr,
                    })
                }
            }
        }

        // Div(u, v) - only if denominator is var-free
        Expr::Div(u, v) => {
            // Var in denominator → non-linear (1/x, x/x, etc.)
            if contains_var(ctx, v, var) {
                return None;
            }

            // u linear, v var-free: (a*x + b) / v = (a/v)*x + (b/v)
            let lf_u = linear_form(ctx, u, var)?;
            let coef = ctx.add(Expr::Div(lf_u.coef, v));
            let constant = ctx.add(Expr::Div(lf_u.constant, v));
            Some(LinearForm { coef, constant })
        }

        // Pow - only linear if exponent is 1 or base is var-free
        Expr::Pow(base, exp) => {
            // If exponent contains var → non-linear
            if contains_var(ctx, exp, var) {
                return None;
            }

            // If base contains var, only linear if exp == 1
            if contains_var(ctx, base, var) {
                if let Expr::Number(n) = ctx.get(exp) {
                    if *n == num_rational::BigRational::from_integer(1.into()) {
                        return linear_form(ctx, base, var);
                    }
                }
                // exp != 1 → non-linear
                return None;
            }

            // Both var-free → constant
            let zero = ctx.num(0);
            Some(LinearForm {
                coef: zero,
                constant: expr,
            })
        }

        // Functions - if any arg contains var, it's non-linear
        Expr::Function(_, args) => {
            if args.iter().any(|&a| contains_var(ctx, a, var)) {
                None // sin(x), log(x), etc. are non-linear
            } else {
                let zero = ctx.num(0);
                Some(LinearForm {
                    coef: zero,
                    constant: expr,
                })
            }
        }

        // Anything else with var → non-linear
        _ => None,
    }
}

/// Try to solve using the structural linear form extractor.
///
/// This is an alternative to the term-based approach that works
/// better for expressions like `y*(1+x)` where the coefficient
/// is itself an expression.
pub(crate) fn try_linear_collect_v2(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<(SolutionSet, Vec<SolveStep>)> {
    // Build expr = lhs - rhs (so expr = 0)
    let expr = simplifier.context.add(Expr::Sub(lhs, rhs));

    // Extract linear form: expr = coef * var + constant = 0
    let lf = linear_form(&mut simplifier.context, expr, var)?;

    // Simplify for cleaner display
    let (coef, _) = simplifier.simplify(lf.coef);
    let (constant, _) = simplifier.simplify(lf.constant);

    // Check if coef contains var (shouldn't, but safety check)
    if contains_var(&simplifier.context, coef, var) {
        return None;
    }

    // Solution: var = -constant / coef  (from coef*var + constant = 0)
    let neg_constant = simplifier.context.add(Expr::Neg(constant));
    let solution = simplifier.context.add(Expr::Div(neg_constant, coef));
    let (solution, _) = simplifier.simplify(solution);

    // Build steps
    let mut steps = Vec::new();

    if simplifier.collect_steps() {
        // Step 1: Show the factored form
        let var_id = simplifier.context.var(var);
        let coef_times_var = simplifier.context.add(Expr::Mul(coef, var_id));
        let factored_lhs = simplifier.context.add(Expr::Add(coef_times_var, constant));
        let zero = simplifier.context.num(0);

        steps.push(SolveStep {
            description: format!("Collect terms in {}", var),
            equation_after: cas_ast::Equation {
                lhs: factored_lhs,
                rhs: zero,
                op: RelOp::Eq,
            },
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });

        // Step 2: Divide by coefficient
        steps.push(SolveStep {
            description: format!(
                "Divide by {}",
                cas_ast::DisplayExpr {
                    context: &simplifier.context,
                    id: coef
                }
            ),
            equation_after: cas_ast::Equation {
                lhs: var_id,
                rhs: solution,
                op: RelOp::Eq,
            },
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    // Check if we can give a direct solution (coef is a ground constant)
    // If coef contains no variables and prove_nonzero returns Proven, skip the Conditional
    if !contains_var(&simplifier.context, coef, var) {
        use crate::domain::Proof;
        use crate::helpers::prove_nonzero;

        let proof = prove_nonzero(&simplifier.context, coef);
        if proof == Proof::Proven {
            // Direct solution: coef is provably non-zero constant
            return Some((SolutionSet::Discrete(vec![solution]), steps));
        }
        // If Disproven (coef == 0), check constant
        if proof == Proof::Disproven {
            let const_proof = prove_nonzero(&simplifier.context, constant);
            if const_proof == Proof::Disproven {
                // Both are 0: 0*x + 0 = 0 → AllReals
                return Some((SolutionSet::AllReals, steps));
            } else if const_proof == Proof::Proven {
                // 0*x + nonzero = 0 → Empty
                return Some((SolutionSet::Empty, steps));
            }
            // Otherwise fall through to Conditional
        }
    }

    // Build conditional solution set
    // Primary case: coef ≠ 0 → {solution}
    let primary_guard = ConditionSet::single(ConditionPredicate::NonZero(coef));
    let primary_case = Case::new(primary_guard, SolutionSet::Discrete(vec![solution]));

    // Degenerate case: coef = 0 ∧ constant = 0 → AllReals
    let mut both_zero_guard = ConditionSet::single(ConditionPredicate::EqZero(coef));
    both_zero_guard.push(ConditionPredicate::EqZero(constant));
    let all_reals_case = Case::new(both_zero_guard, SolutionSet::AllReals);

    // Otherwise → Empty
    let otherwise_case = Case::new(ConditionSet::empty(), SolutionSet::Empty);

    let solution_set = SolutionSet::Conditional(vec![primary_case, all_reals_case, otherwise_case]);

    Some((solution_set, steps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_terms_signed() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");

        // a + b + c
        let ab = ctx.add(Expr::Add(a, b));
        let abc = ctx.add(Expr::Add(ab, c));

        let terms = add_terms_signed(&ctx, abc);
        assert_eq!(terms.len(), 3);
    }

    #[test]
    fn test_split_linear_term_const() {
        let mut ctx = Context::new();
        let a = ctx.var("A");

        match split_linear_term(&mut ctx, a, "P") {
            TermClass::Const(_) => {}
            _ => panic!("A should be Const with respect to P"),
        }
    }

    #[test]
    fn test_split_linear_term_var() {
        let mut ctx = Context::new();
        let p = ctx.var("P");

        match split_linear_term(&mut ctx, p, "P") {
            TermClass::Linear(_) => {}
            _ => panic!("P should be Linear(1) with respect to P"),
        }
    }
}
