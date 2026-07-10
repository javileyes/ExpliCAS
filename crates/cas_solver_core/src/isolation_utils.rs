use crate::solution_set::{intersect_solution_sets, union_solution_sets};
use cas_ast::{Context, Expr, ExprId, RelOp, SolutionSet};
use std::cmp::Ordering;

/// Create a residual solve expression: solve(lhs ⟨op⟩ rhs, var).
/// Used when the solver can't justify a step but wants graceful degradation.
/// The ORIGINAL relational operator is preserved (scout cycle-3 honesty
/// contract: `solve(sin(x) > 1/2)` must not echo back as `sin(x) = 1/2`);
/// equations keep the canonical `__eq__` wrapper, inequalities lower to the
/// symbolic-comparison builtins (`Less`, `GreaterEqual`, …) the formatter
/// renders as `a < b`.
pub(crate) fn mk_residual_solve(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
) -> ExprId {
    let rel_expr = if matches!(op, RelOp::Eq) {
        cas_ast::eq::wrap_eq(ctx, lhs, rhs)
    } else {
        ctx.call(op.builtin_name(), vec![lhs, rhs])
    };
    let var_expr = ctx.var(var);
    ctx.call("solve", vec![rel_expr, var_expr])
}

/// Check whether an expression contains a specific named variable.
pub fn contains_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Variable(sym_id) => ctx.sym_name(*sym_id) == var,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            contains_var(ctx, *l, var) || contains_var(ctx, *r, var)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_var(ctx, *inner, var),
        Expr::Function(_, args) => args.iter().any(|&arg| contains_var(ctx, arg, var)),
        Expr::Matrix { data, .. } => data.iter().any(|&elem| contains_var(ctx, elem, var)),
        Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
    }
}

/// Check if an expression is known to be negative.
///
/// Recursively analyzes Mul products using XOR logic:
/// `(-a) * b` is negative, `(-a) * (-b)` is positive.
pub(crate) fn is_known_negative(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n < num_rational::BigRational::from_integer(0.into()),
        Expr::Neg(_) => true,
        Expr::Mul(l, r) => is_known_negative(ctx, *l) ^ is_known_negative(ctx, *r),
        // A constant LINEAR SURD `A + B·√n` (e.g. `1 − √2`): decide its sign exactly, so an even-root
        // isolation `x² = 1 − √2` correctly drops to No solution instead of leaking `±√(1−√2)`.
        // Fallback: the exact value-bounds oracle decides general constants the surd
        // form misses (`1 − e^(1/3)`, `1 − π`), same discipline, never a guess.
        _ => {
            cas_math::root_forms::provable_sign_vs_zero(ctx, expr) == Some(Ordering::Less)
                || cas_math::const_sign::provable_const_sign(ctx, expr)
                    == Some(cas_math::const_sign::ConstSign::Negative)
        }
    }
}

/// Attempt to recompose a^e / b^e -> (a/b)^e when both powers have the same exponent.
pub(crate) fn try_recompose_pow_quotient(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Div(num, den) = expr_data {
        let num_data = ctx.get(num).clone();
        let den_data = ctx.get(den).clone();
        if let (Expr::Pow(a, e1), Expr::Pow(b, e2)) = (num_data, den_data) {
            if cas_ast::ordering::compare_expr(ctx, e1, e2) == Ordering::Equal {
                let new_base = ctx.add(Expr::Div(a, b));
                return Some(ctx.add(Expr::Pow(new_base, e1)));
            }
        }
    }
    None
}

/// Flip inequality direction under multiplication/division by a negative value.
pub fn flip_inequality(op: RelOp) -> RelOp {
    match op {
        RelOp::Eq => RelOp::Eq,
        RelOp::Neq => RelOp::Neq,
        RelOp::Lt => RelOp::Gt,
        RelOp::Gt => RelOp::Lt,
        RelOp::Leq => RelOp::Geq,
        RelOp::Geq => RelOp::Leq,
    }
}

/// Check if expr is `1/var` pattern (simple reciprocal of target variable).
pub(crate) fn is_simple_reciprocal(ctx: &Context, expr: ExprId, var: &str) -> bool {
    if let Expr::Div(num, denom) = ctx.get(expr) {
        let is_one = matches!(
            ctx.get(*num),
            Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
        );
        let is_var =
            matches!(ctx.get(*denom), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var);
        is_one && is_var
    } else {
        false
    }
}

/// True iff expression is exactly the target variable.
pub(crate) fn is_target_variable(ctx: &Context, expr: ExprId, var: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var)
}

/// True iff `left * right op 0` should branch into product sign cases.
pub(crate) fn should_split_product_zero_inequality(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
    rhs: ExprId,
    op: &RelOp,
    var: &str,
) -> bool {
    contains_var(ctx, left, var)
        && contains_var(ctx, right, var)
        && is_numeric_zero(ctx, rhs)
        && is_inequality_relop(op)
}

/// True iff `numerator/denominator op rhs` should split by denominator sign.
///
/// The split is valid whenever the solve variable is in the DENOMINATOR and the
/// relation is an inequality — cross-multiplying by the denominator's signed value
/// is sound regardless of whether the (var-free or var-bearing) numerator also
/// contains the variable. This is only ever consulted from the
/// `VariableInNumerator` route, where `contains_var(numerator)` is already
/// guaranteed, so omitting that redundant check is behaviour-preserving there
/// while also licensing the Cluster E constant-numerator case `1/(x-2) > 1`
/// (deliberately routed through the numerator pipeline for the split).
pub(crate) fn should_split_division_denominator_sign_cases(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
    op: &RelOp,
    var: &str,
) -> bool {
    let _ = numerator;
    contains_var(ctx, denominator, var) && is_inequality_relop(op)
}

/// True iff already-isolated denominator variable `x op rhs` should split
/// into `x > 0` and `x < 0` branches.
pub(crate) fn should_split_isolated_denominator_variable(
    ctx: &Context,
    denominator: ExprId,
    op: &RelOp,
    var: &str,
) -> bool {
    is_target_variable(ctx, denominator, var) && is_inequality_relop(op)
}

/// True iff reciprocal solve strategy should be attempted for current equation side.
pub(crate) fn should_try_reciprocal_solve(
    ctx: &Context,
    lhs: ExprId,
    op: &RelOp,
    var: &str,
) -> bool {
    matches!(op, RelOp::Eq) && is_simple_reciprocal(ctx, lhs, var)
}

/// True iff expression is the numeric literal zero.
pub(crate) fn is_numeric_zero(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(n) if *n == num_rational::BigRational::from_integer(0.into())
    )
}

/// True iff expression is the numeric literal one.
pub fn is_numeric_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
    )
}

/// Trinary numeric sign classification for literal numeric expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericSign {
    Negative,
    Zero,
    Positive,
}

/// Return sign for numeric literal expressions, or `None` for non-numeric nodes.
pub(crate) fn numeric_sign(ctx: &Context, expr: ExprId) -> Option<NumericSign> {
    let Expr::Number(n) = ctx.get(expr) else {
        return None;
    };
    let zero = num_rational::BigRational::from_integer(0.into());
    if *n < zero {
        Some(NumericSign::Negative)
    } else if *n == zero {
        Some(NumericSign::Zero)
    } else {
        Some(NumericSign::Positive)
    }
}

/// Sign of a provably-constant expression for isolation decisions.
///
/// Fast-paths numeric literals via [`numeric_sign`], then consults the EXACT
/// constant oracles: `root_forms::provable_sign_vs_zero` (linear surds `A + B·√n`)
/// and `const_sign::provable_const_sign` (transcendentals `ln(1/2)`, `e − 3`).
/// Returns `None` when the sign cannot be proven — callers must treat that as
/// undecided, never as "non-negative".
pub(crate) fn const_numeric_sign(ctx: &Context, expr: ExprId) -> Option<NumericSign> {
    if let Some(sign) = numeric_sign(ctx, expr) {
        return Some(sign);
    }
    if let Some(ord) = cas_math::root_forms::provable_sign_vs_zero(ctx, expr) {
        return Some(match ord {
            Ordering::Less => NumericSign::Negative,
            Ordering::Equal => NumericSign::Zero,
            Ordering::Greater => NumericSign::Positive,
        });
    }
    match cas_math::const_sign::provable_const_sign(ctx, expr)? {
        cas_math::const_sign::ConstSign::Negative => Some(NumericSign::Negative),
        cas_math::const_sign::ConstSign::Zero => Some(NumericSign::Zero),
        cas_math::const_sign::ConstSign::Positive => Some(NumericSign::Positive),
    }
}

/// True iff expression is a numeric even integer literal.
pub(crate) fn is_even_integer_expr(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => n.is_integer() && (n.to_integer() % 2 == 0.into()),
        _ => false,
    }
}

/// True iff expression denotes a positive integer numeric literal.
///
/// Accepts either:
/// - `Number(n)` where `n` is an integer > 0
/// - `Div(Number(n), Number(d))` that evaluates to an integer > 0
pub(crate) fn is_positive_integer_expr(ctx: &Context, expr: ExprId) -> bool {
    let zero = num_rational::BigRational::from_integer(0.into());
    match ctx.get(expr) {
        Expr::Number(n) => n.is_integer() && *n > zero,
        Expr::Div(n_id, d_id) => {
            if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*n_id), ctx.get(*d_id)) {
                if *d == zero {
                    return false;
                }
                let val = n / d;
                val.is_integer() && val > zero
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Extract factors for zero-product splitting:
/// - `A*B` -> `[A, B]`
/// - `A^n` with positive-integer `n` -> `[A]`
pub(crate) fn split_zero_product_factors(ctx: &Context, expr: ExprId) -> Option<Vec<ExprId>> {
    match ctx.get(expr) {
        Expr::Mul(l, r) => Some(vec![*l, *r]),
        Expr::Pow(base, exp) if is_positive_integer_expr(ctx, *exp) => Some(vec![*base]),
        _ => None,
    }
}

/// Matched exponential expression `base^exponent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExponentialPattern {
    pub base: ExprId,
    pub exponent: ExprId,
}

/// Exponential candidate for equations where the variable appears on only
/// one side as an exponent (`a^x = b` or `b = a^x`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SingleSideExponentialCandidate {
    pub base: ExprId,
    pub other_side: ExprId,
}

/// Match `base^exponent` where `base` contains `var` and exponent does not.
pub(crate) fn match_exponential_var_in_base(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<ExponentialPattern> {
    if let Expr::Pow(base, exponent) = ctx.get(expr) {
        if contains_var(ctx, *base, var) && !contains_var(ctx, *exponent, var) {
            return Some(ExponentialPattern {
                base: *base,
                exponent: *exponent,
            });
        }
    }
    None
}

/// Match `base^exponent` where exponent contains `var` and base does not.
pub fn match_exponential_var_in_exponent(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<ExponentialPattern> {
    if let Expr::Pow(base, exponent) = ctx.get(expr) {
        if contains_var(ctx, *exponent, var) && !contains_var(ctx, *base, var) {
            return Some(ExponentialPattern {
                base: *base,
                exponent: *exponent,
            });
        }
    }
    None
}

/// Find `base^(...var...)` on exactly one side of an equation.
///
/// `lhs_has_var` and `rhs_has_var` are provided by the caller to avoid
/// duplicate containment scans in strategy pipelines.
pub(crate) fn find_single_side_exponential_var_in_exponent(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    lhs_has_var: bool,
    rhs_has_var: bool,
) -> Option<SingleSideExponentialCandidate> {
    if lhs_has_var && !rhs_has_var {
        let pattern = match_exponential_var_in_exponent(ctx, lhs, var)?;
        Some(SingleSideExponentialCandidate {
            base: pattern.base,
            other_side: rhs,
        })
    } else if rhs_has_var && !lhs_has_var {
        let pattern = match_exponential_var_in_exponent(ctx, rhs, var)?;
        Some(SingleSideExponentialCandidate {
            base: pattern.base,
            other_side: lhs,
        })
    } else {
        None
    }
}

/// Combine the two branch solution sets generated from `|A| op B`.
///
/// For equalities/greater-than forms both branches are alternatives (union).
/// For less-than forms both constraints must hold simultaneously (intersection).
pub(crate) fn combine_abs_branch_sets(
    ctx: &Context,
    op: RelOp,
    positive_branch: SolutionSet,
    negative_branch: SolutionSet,
) -> SolutionSet {
    match op {
        RelOp::Eq | RelOp::Neq | RelOp::Gt | RelOp::Geq => {
            union_solution_sets(ctx, positive_branch, negative_branch)
        }
        RelOp::Lt | RelOp::Leq => intersect_solution_sets(ctx, positive_branch, negative_branch),
    }
}

/// Relational operators to apply to each factor in a product-sign split case.
#[derive(Debug, Clone, PartialEq)]
pub struct SignCaseOps {
    pub left: RelOp,
    pub right: RelOp,
}

/// Build the two sign cases for product inequalities with zero RHS:
/// `A*B op 0`.
///
/// Returns `None` for non-inequality operators.
pub(crate) fn product_zero_inequality_cases(op: RelOp) -> Option<(SignCaseOps, SignCaseOps)> {
    match op {
        RelOp::Gt => Some((
            SignCaseOps {
                left: RelOp::Gt,
                right: RelOp::Gt,
            },
            SignCaseOps {
                left: RelOp::Lt,
                right: RelOp::Lt,
            },
        )),
        RelOp::Geq => Some((
            SignCaseOps {
                left: RelOp::Geq,
                right: RelOp::Geq,
            },
            SignCaseOps {
                left: RelOp::Leq,
                right: RelOp::Leq,
            },
        )),
        RelOp::Lt => Some((
            SignCaseOps {
                left: RelOp::Gt,
                right: RelOp::Lt,
            },
            SignCaseOps {
                left: RelOp::Lt,
                right: RelOp::Gt,
            },
        )),
        RelOp::Leq => Some((
            SignCaseOps {
                left: RelOp::Geq,
                right: RelOp::Leq,
            },
            SignCaseOps {
                left: RelOp::Leq,
                right: RelOp::Geq,
            },
        )),
        RelOp::Eq | RelOp::Neq => None,
    }
}

/// For inequality `A / B op C`, returns operators for denominator sign split:
/// `(op_when_B_positive, op_when_B_negative)`.
pub(crate) fn denominator_sign_case_ops(op: RelOp) -> Option<(RelOp, RelOp)> {
    if is_inequality_relop(&op) {
        Some((op.clone(), flip_inequality(op)))
    } else {
        None
    }
}

/// For `A / x op B` while isolating denominator variable `x` directly,
/// returns operators `(op_when_x_positive, op_when_x_negative)`.
///
/// This is the inverse perspective of [`denominator_sign_case_ops`], so
/// the pair is intentionally swapped.
pub(crate) fn isolated_denominator_variable_case_ops(op: RelOp) -> Option<(RelOp, RelOp)> {
    denominator_sign_case_ops(op).map(|(op_when_pos, op_when_neg)| (op_when_neg, op_when_pos))
}

/// True iff relation operator is an inequality (`<`, `>`, `<=`, `>=`).
pub(crate) fn is_inequality_relop(op: &RelOp) -> bool {
    matches!(op, RelOp::Lt | RelOp::Gt | RelOp::Leq | RelOp::Geq)
}

/// Route a single-variable polynomial INEQUALITY of degree >= 3 to its FACTORED form, so it flows
/// through the existing product-sign path (`should_split_product_zero_inequality`) instead of the
/// variable-isolation strategy, which mis-handles it: `solve(x^3-x<0)` produced a garbled
/// `solve(x = x^(1/3))` and `solve(x^4-5x^2+4<0)` returned the empty set, while the equivalent
/// factored inputs solve correctly.
///
/// Returns the factored equation `factor(lhs-rhs) OP 0`, or `None` when it does not apply:
/// not an inequality; not univariate in `var`; degree < 3 (quadratics already solve numerically);
/// the difference is ALREADY a product (the product path handles it — also the loop guard for our
/// own factored re-entry); or the polynomial is irreducible / not improved by factoring (left to
/// the existing path, e.g. `x^4-10` stays an exact `(-10^(1/4), 10^(1/4))`). EXACT: reuses the
/// `BigRational`/`BigInt` polynomial factorer — never an `f64` keep/drop.
pub(crate) fn try_factor_polynomial_inequality(
    ctx: &mut Context,
    eq: &cas_ast::Equation,
    var: &str,
) -> Option<cas_ast::Equation> {
    use num_traits::Zero;
    if !is_inequality_relop(&eq.op) {
        return None;
    }
    let rhs_is_zero = matches!(ctx.get(eq.rhs), Expr::Number(n) if n.is_zero());
    let diff_raw = if rhs_is_zero {
        eq.lhs
    } else {
        ctx.add(Expr::Sub(eq.lhs, eq.rhs))
    };
    // Already a product (user-factored, or our own factored re-entry): the product-sign path
    // handles it. Checking the RAW form — NOT the expanded one — is what prevents an
    // expand -> factor -> expand loop.
    if matches!(ctx.get(diff_raw), Expr::Mul(_, _)) {
        return None;
    }
    let diff = cas_math::expand_ops::expand(ctx, diff_raw);
    // Univariate in `var` only (a foreign variable would make the product inequality multivariate).
    let vars = cas_ast::collect_variables(ctx, diff);
    if vars.len() != 1 || !vars.contains(var) {
        return None;
    }
    // Degree >= 3: quadratics already solve via the numeric quadratic path; leave them alone.
    let poly = cas_math::polynomial::Polynomial::from_expr(ctx, diff, var).ok()?;
    if poly.degree() < 3 {
        return None;
    }
    let factored = cas_math::factor::factor(ctx, diff);
    if !matches!(ctx.get(factored), Expr::Mul(_, _)) {
        return None; // irreducible / not improved -> leave to the existing isolation path
    }
    let zero = ctx.num(0);
    Some(cas_ast::Equation {
        lhs: factored,
        rhs: zero,
        op: eq.op.clone(),
    })
}

/// Flip inequality only when multiplying/dividing by a known negative term.
pub(crate) fn apply_sign_flip(op: RelOp, known_negative: bool) -> RelOp {
    if known_negative {
        flip_inequality(op)
    } else {
        op
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{BoundType, Expr, Interval};

    #[test]
    fn const_numeric_sign_decides_literals_surds_and_transcendentals() {
        use cas_parser::parse;
        let mut ctx = Context::new();
        let cases = [
            ("-2", Some(NumericSign::Negative)),
            ("0", Some(NumericSign::Zero)),
            ("3/2", Some(NumericSign::Positive)),
            ("sqrt(2) - 2", Some(NumericSign::Negative)),
            ("2 - sqrt(2)", Some(NumericSign::Positive)),
            ("ln(1/2)", Some(NumericSign::Negative)),
            ("-ln(2)", Some(NumericSign::Negative)),
            ("e - 3", Some(NumericSign::Negative)),
            ("ln(2)", Some(NumericSign::Positive)),
            // Not provably constant: must stay undecided, never guessed.
            ("y", None),
            ("y - 1", None),
        ];
        for (src, expected) in cases {
            let expr = parse(src, &mut ctx).expect("parse");
            assert_eq!(
                const_numeric_sign(&ctx, expr),
                expected,
                "const_numeric_sign({src})"
            );
        }
    }

    #[test]
    fn factor_polynomial_inequality_applies_only_to_reducible_higher_degree() {
        use cas_parser::parse;
        let ineq = |ctx: &mut Context, lhs_src: &str, op: RelOp| -> cas_ast::Equation {
            let lhs = parse(lhs_src, ctx).expect("parse lhs");
            let rhs = ctx.num(0);
            cas_ast::Equation { lhs, rhs, op }
        };
        // Reducible degree >= 3 inequality -> factored to a product (Mul), so the product-sign
        // path can handle it.
        {
            let mut ctx = Context::new();
            let eq = ineq(&mut ctx, "x^3 - x", RelOp::Lt);
            let out = try_factor_polynomial_inequality(&mut ctx, &eq, "x")
                .expect("x^3-x<0 should factor");
            assert!(
                matches!(ctx.get(out.lhs), Expr::Mul(_, _)),
                "factored lhs is a product"
            );
            assert_eq!(out.op, RelOp::Lt);
        }
        // Declines: degree 2 (numeric quadratic path), irreducible quartic, already-product,
        // and an EQUATION (not an inequality).
        for (src, op) in [
            ("x^2 - 3*x + 2", RelOp::Lt),     // degree 2
            ("x^4 - 10", RelOp::Lt),          // irreducible over Q
            ("(x-1)*(x-2)*(x-3)", RelOp::Lt), // already a product
            ("x^3 - x", RelOp::Eq),           // equation, not inequality
        ] {
            let mut ctx = Context::new();
            let eq = ineq(&mut ctx, src, op.clone());
            assert!(
                try_factor_polynomial_inequality(&mut ctx, &eq, "x").is_none(),
                "{src} ({op:?}) must decline"
            );
        }
    }

    #[test]
    fn test_is_simple_reciprocal() {
        let mut ctx = Context::new();
        let r = ctx.var("R");
        let one = ctx.num(1);
        let reciprocal = ctx.add(Expr::Div(one, r));

        assert!(is_simple_reciprocal(&ctx, reciprocal, "R"));
        assert!(!is_simple_reciprocal(&ctx, reciprocal, "X"));
        assert!(!is_simple_reciprocal(&ctx, r, "R"));
    }

    #[test]
    fn test_is_target_variable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        assert!(is_target_variable(&ctx, x, "x"));
        assert!(!is_target_variable(&ctx, y, "x"));
    }

    #[test]
    fn test_should_split_product_zero_inequality() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_one = ctx.add(Expr::Add(x, one));
        let zero = ctx.num(0);

        assert!(should_split_product_zero_inequality(
            &ctx,
            x,
            x_plus_one,
            zero,
            &RelOp::Lt,
            "x"
        ));
        assert!(!should_split_product_zero_inequality(
            &ctx,
            x,
            x_plus_one,
            zero,
            &RelOp::Eq,
            "x"
        ));
    }

    #[test]
    fn test_should_split_division_denominator_sign_cases() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_one = ctx.add(Expr::Add(x, one));

        assert!(should_split_division_denominator_sign_cases(
            &ctx,
            x,
            x_plus_one,
            &RelOp::Leq,
            "x"
        ));
        assert!(!should_split_division_denominator_sign_cases(
            &ctx,
            x,
            x_plus_one,
            &RelOp::Eq,
            "x"
        ));
        // Cluster E: a CONSTANT numerator over a var-bearing denominator must also
        // split (e.g. `1/(x-2) > 1`) -- the var is in the denominator.
        assert!(should_split_division_denominator_sign_cases(
            &ctx,
            one,
            x_plus_one,
            &RelOp::Gt,
            "x"
        ));
        // No variable in the denominator -> no split.
        assert!(!should_split_division_denominator_sign_cases(
            &ctx,
            x,
            one,
            &RelOp::Gt,
            "x"
        ));
    }

    #[test]
    fn test_should_split_isolated_denominator_variable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        assert!(should_split_isolated_denominator_variable(
            &ctx,
            x,
            &RelOp::Gt,
            "x"
        ));
        assert!(!should_split_isolated_denominator_variable(
            &ctx,
            y,
            &RelOp::Gt,
            "x"
        ));
        assert!(!should_split_isolated_denominator_variable(
            &ctx,
            x,
            &RelOp::Eq,
            "x"
        ));
    }

    #[test]
    fn test_should_try_reciprocal_solve() {
        let mut ctx = Context::new();
        let r = ctx.var("R");
        let one = ctx.num(1);
        let reciprocal = ctx.add(Expr::Div(one, r));

        assert!(should_try_reciprocal_solve(
            &ctx,
            reciprocal,
            &RelOp::Eq,
            "R"
        ));
        assert!(!should_try_reciprocal_solve(
            &ctx,
            reciprocal,
            &RelOp::Leq,
            "R"
        ));
    }

    #[test]
    fn test_combine_abs_branch_sets_union_for_eq() {
        let mut ctx = Context::new();
        let i1 = Interval {
            min: ctx.num(0),
            min_type: BoundType::Closed,
            max: ctx.num(1),
            max_type: BoundType::Closed,
        };
        let i2 = Interval {
            min: ctx.num(2),
            min_type: BoundType::Closed,
            max: ctx.num(3),
            max_type: BoundType::Closed,
        };

        let set = combine_abs_branch_sets(
            &ctx,
            RelOp::Eq,
            SolutionSet::Continuous(i1),
            SolutionSet::Continuous(i2),
        );
        assert!(matches!(set, SolutionSet::Union(v) if v.len() == 2));
    }

    #[test]
    fn test_combine_abs_branch_sets_intersection_for_lt() {
        let mut ctx = Context::new();
        let i1 = Interval {
            min: ctx.num(0),
            min_type: BoundType::Closed,
            max: ctx.num(2),
            max_type: BoundType::Closed,
        };
        let i2 = Interval {
            min: ctx.num(1),
            min_type: BoundType::Closed,
            max: ctx.num(3),
            max_type: BoundType::Closed,
        };

        let set = combine_abs_branch_sets(
            &ctx,
            RelOp::Lt,
            SolutionSet::Continuous(i1),
            SolutionSet::Continuous(i2),
        );
        match set {
            SolutionSet::Continuous(i) => {
                assert_eq!(i.min, ctx.num(1));
                assert_eq!(i.max, ctx.num(2));
            }
            other => panic!("Expected Continuous intersection, got {:?}", other),
        }
    }

    #[test]
    fn test_product_zero_inequality_cases_gt() {
        let (c1, c2) = product_zero_inequality_cases(RelOp::Gt).expect("expected cases");
        assert_eq!(
            c1,
            SignCaseOps {
                left: RelOp::Gt,
                right: RelOp::Gt
            }
        );
        assert_eq!(
            c2,
            SignCaseOps {
                left: RelOp::Lt,
                right: RelOp::Lt
            }
        );
    }

    #[test]
    fn test_product_zero_inequality_cases_leq() {
        let (c1, c2) = product_zero_inequality_cases(RelOp::Leq).expect("expected cases");
        assert_eq!(
            c1,
            SignCaseOps {
                left: RelOp::Geq,
                right: RelOp::Leq
            }
        );
        assert_eq!(
            c2,
            SignCaseOps {
                left: RelOp::Leq,
                right: RelOp::Geq
            }
        );
    }

    #[test]
    fn test_product_zero_inequality_cases_eq_none() {
        assert!(product_zero_inequality_cases(RelOp::Eq).is_none());
        assert!(product_zero_inequality_cases(RelOp::Neq).is_none());
    }

    #[test]
    fn test_is_numeric_zero() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let two = ctx.num(2);
        assert!(is_numeric_zero(&ctx, zero));
        assert!(!is_numeric_zero(&ctx, two));
    }

    #[test]
    fn test_is_numeric_one() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        assert!(is_numeric_one(&ctx, one));
        assert!(!is_numeric_one(&ctx, two));
    }

    #[test]
    fn test_numeric_sign() {
        let mut ctx = Context::new();
        let neg = ctx.num(-3);
        let zero = ctx.num(0);
        let pos = ctx.num(5);
        let sym = ctx.var("x");

        assert_eq!(numeric_sign(&ctx, neg), Some(NumericSign::Negative));
        assert_eq!(numeric_sign(&ctx, zero), Some(NumericSign::Zero));
        assert_eq!(numeric_sign(&ctx, pos), Some(NumericSign::Positive));
        assert_eq!(numeric_sign(&ctx, sym), None);
    }

    #[test]
    fn test_is_positive_integer_expr() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let neg_two = ctx.num(-2);
        let three = ctx.num(3);
        let six = ctx.num(6);
        let one = ctx.num(1);
        let two_den = ctx.num(2);
        let half = ctx.add(Expr::Div(one, two_den));
        let two_from_div = ctx.add(Expr::Div(six, three));
        let one2 = ctx.num(1);
        let zero = ctx.num(0);
        let div_zero = ctx.add(Expr::Div(one2, zero));
        let x = ctx.var("x");

        assert!(is_positive_integer_expr(&ctx, two));
        assert!(!is_positive_integer_expr(&ctx, neg_two));
        assert!(is_positive_integer_expr(&ctx, two_from_div));
        assert!(!is_positive_integer_expr(&ctx, half));
        assert!(!is_positive_integer_expr(&ctx, div_zero));
        assert!(!is_positive_integer_expr(&ctx, x));
    }

    #[test]
    fn test_split_zero_product_factors_mul() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let mul = ctx.add(Expr::Mul(a, b));
        let factors = split_zero_product_factors(&ctx, mul).expect("expected factors");
        assert_eq!(factors, vec![a, b]);
    }

    #[test]
    fn test_split_zero_product_factors_pow_positive_int() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let pow = ctx.add(Expr::Pow(x, two));
        let factors = split_zero_product_factors(&ctx, pow).expect("expected factors");
        assert_eq!(factors, vec![x]);
    }

    #[test]
    fn test_split_zero_product_factors_pow_non_positive_int() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let pow = ctx.add(Expr::Pow(x, zero));
        assert!(split_zero_product_factors(&ctx, pow).is_none());
    }

    #[test]
    fn test_match_exponential_var_in_base() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Pow(x, two));
        let m = match_exponential_var_in_base(&ctx, expr, "x").expect("must match");
        assert_eq!(m.base, x);
        assert_eq!(m.exponent, two);
    }

    #[test]
    fn test_match_exponential_var_in_exponent() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        let expr = ctx.add(Expr::Pow(two, x));
        let m = match_exponential_var_in_exponent(&ctx, expr, "x").expect("must match");
        assert_eq!(m.base, two);
        assert_eq!(m.exponent, x);
    }

    #[test]
    fn test_find_single_side_exponential_var_in_exponent_lhs() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        let y = ctx.var("y");
        let lhs = ctx.add(Expr::Pow(two, x));
        let rhs = y;
        let c = find_single_side_exponential_var_in_exponent(&ctx, lhs, rhs, "x", true, false)
            .expect("must match lhs candidate");
        assert_eq!(c.base, two);
        assert_eq!(c.other_side, y);
    }

    #[test]
    fn test_find_single_side_exponential_var_in_exponent_rhs() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        let y = ctx.var("y");
        let lhs = y;
        let rhs = ctx.add(Expr::Pow(two, x));
        let c = find_single_side_exponential_var_in_exponent(&ctx, lhs, rhs, "x", false, true)
            .expect("must match rhs candidate");
        assert_eq!(c.base, two);
        assert_eq!(c.other_side, y);
    }

    #[test]
    fn test_find_single_side_exponential_var_in_exponent_rejects_both_sides() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        let lhs = ctx.add(Expr::Pow(two, x));
        let rhs = ctx.add(Expr::Pow(two, x));
        assert!(
            find_single_side_exponential_var_in_exponent(&ctx, lhs, rhs, "x", true, true).is_none()
        );
    }

    #[test]
    fn test_is_even_integer_expr() {
        let mut ctx = Context::new();
        let four = ctx.num(4);
        let three = ctx.num(3);
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        assert!(is_even_integer_expr(&ctx, four));
        assert!(!is_even_integer_expr(&ctx, three));
        assert!(!is_even_integer_expr(&ctx, half));
    }

    #[test]
    fn test_denominator_sign_case_ops() {
        let (pos, neg) = denominator_sign_case_ops(RelOp::Leq).expect("expected cases");
        assert_eq!(pos, RelOp::Leq);
        assert_eq!(neg, RelOp::Geq);
        assert!(denominator_sign_case_ops(RelOp::Eq).is_none());
    }

    #[test]
    fn test_isolated_denominator_variable_case_ops() {
        let (pos, neg) =
            isolated_denominator_variable_case_ops(RelOp::Leq).expect("expected cases");
        assert_eq!(pos, RelOp::Geq);
        assert_eq!(neg, RelOp::Leq);
        assert!(isolated_denominator_variable_case_ops(RelOp::Eq).is_none());
    }

    #[test]
    fn test_is_inequality_relop() {
        assert!(is_inequality_relop(&RelOp::Lt));
        assert!(is_inequality_relop(&RelOp::Geq));
        assert!(!is_inequality_relop(&RelOp::Eq));
        assert!(!is_inequality_relop(&RelOp::Neq));
    }

    #[test]
    fn test_apply_sign_flip() {
        assert_eq!(apply_sign_flip(RelOp::Gt, true), RelOp::Lt);
        assert_eq!(apply_sign_flip(RelOp::Gt, false), RelOp::Gt);
        assert_eq!(apply_sign_flip(RelOp::Eq, true), RelOp::Eq);
    }
}
