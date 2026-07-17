//! Implicit-domain inference and domain-delta checks.
//!
//! This module is runtime-agnostic. Runtime crates provide:
//! - whether inference is enabled (`real_only`)
//! - positivity prover callback
//! - domain inference callback when comparing rewrites

use crate::domain_condition::{ImplicitCondition, ImplicitDomain};
use crate::domain_proof::Proof;
use crate::domain_witness::{witness_survives_in_context, WitnessKind};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_extract::{
    extract_abs_argument_view, extract_log_base_argument_view, extract_sqrt_argument_view,
    extract_unary_log_argument_view, log10_base_sentinel,
};
use cas_math::expr_predicates::{
    contains_variable, has_positivity_structure, is_even_root_exponent,
};
use cas_math::numeric_eval::as_rational_const;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

/// Result of domain delta check between input and output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomainDelta {
    /// No domain expansion (output has same or more constraints).
    Safe,
    /// Domain was expanded by removing Analytic constraints (x>=0, x>0).
    ExpandsAnalytic(Vec<ImplicitCondition>),
    /// Domain was expanded by removing Definability constraints (x!=0).
    ExpandsDefinability(Vec<ImplicitCondition>),
}

/// Result of checking if a rewrite would expand analytic domain.
#[derive(Debug, Clone)]
pub enum AnalyticExpansionResult {
    /// No expansion - rewrite is safe.
    Safe,
    /// Would expand domain - contains predicates whose witnesses don't survive.
    /// Fields: (dropped_predicates, source_descriptions).
    WouldExpand {
        dropped: Vec<ImplicitCondition>,
        sources: Vec<String>, // e.g., "x >= 0 (from sqrt(x))"
    },
}

impl AnalyticExpansionResult {
    pub fn is_safe(&self) -> bool {
        matches!(self, AnalyticExpansionResult::Safe)
    }

    pub fn would_expand(&self) -> bool {
        matches!(self, AnalyticExpansionResult::WouldExpand { .. })
    }
}

/// Check if a rewrite would expand the domain by removing implicit constraints.
///
/// Callers provide the implicit-domain inference function so this helper remains
/// decoupled from runtime-specific value-domain enums and counters.
pub fn domain_delta_check<F>(
    ctx: &Context,
    input: ExprId,
    output: ExprId,
    mut infer_domain: F,
) -> DomainDelta
where
    F: FnMut(&Context, ExprId) -> ImplicitDomain,
{
    let d_in = infer_domain(ctx, input);
    let d_out = infer_domain(ctx, output);

    let dropped: Vec<ImplicitCondition> = d_in
        .conditions()
        .iter()
        .filter(|c| !d_out.conditions().contains(c))
        .cloned()
        .collect();

    if dropped.is_empty() {
        return DomainDelta::Safe;
    }

    let analytic: Vec<ImplicitCondition> = dropped
        .iter()
        .filter(|c| {
            matches!(
                c,
                ImplicitCondition::NonNegative(_)
                    | ImplicitCondition::LowerBound(_, _)
                    | ImplicitCondition::Positive(_)
            )
        })
        .cloned()
        .collect();

    let definability: Vec<ImplicitCondition> = dropped
        .iter()
        .filter(|c| matches!(c, ImplicitCondition::NonZero(_)))
        .cloned()
        .collect();

    if !analytic.is_empty() {
        DomainDelta::ExpandsAnalytic(analytic)
    } else if !definability.is_empty() {
        DomainDelta::ExpandsDefinability(definability)
    } else {
        DomainDelta::Safe
    }
}

/// Quick check: does this rewrite expand analytic domain?
pub fn expands_analytic_domain<F>(
    ctx: &Context,
    input: ExprId,
    output: ExprId,
    infer_domain: F,
) -> bool
where
    F: FnMut(&Context, ExprId) -> ImplicitDomain,
{
    matches!(
        domain_delta_check(ctx, input, output, infer_domain),
        DomainDelta::ExpandsAnalytic(_)
    )
}

/// Context-aware check: does this rewrite expand analytic domain considering
/// the full tree and surviving witnesses.
pub fn check_analytic_expansion<F>(
    ctx: &Context,
    root: ExprId,
    rewritten_node: ExprId,
    replacement: ExprId,
    infer_domain: F,
) -> AnalyticExpansionResult
where
    F: FnMut(&Context, ExprId) -> ImplicitDomain,
{
    let delta = domain_delta_check(ctx, rewritten_node, replacement, infer_domain);

    match delta {
        DomainDelta::Safe => AnalyticExpansionResult::Safe,
        DomainDelta::ExpandsDefinability(_) => AnalyticExpansionResult::Safe,
        DomainDelta::ExpandsAnalytic(dropped) => {
            let mut unsatisfied: Vec<ImplicitCondition> = Vec::new();
            let mut sources: Vec<String> = Vec::new();

            for cond in dropped {
                let (target, kind, predicate_str, source_str) = match &cond {
                    // A branch annotation is not an analytic-domain
                    // predicate: it never comes from domain inference and
                    // cannot be "unsatisfied" by a real-domain rewrite.
                    ImplicitCondition::PrincipalBranch { .. } => continue,
                    ImplicitCondition::NonNegative(t) => {
                        let var_name = format_expr_short(ctx, *t);
                        (
                            *t,
                            WitnessKind::Sqrt,
                            format!("{} >= 0", var_name),
                            format!("from sqrt({})", var_name),
                        )
                    }
                    ImplicitCondition::Positive(t) => {
                        let var_name = format_expr_short(ctx, *t);
                        (
                            *t,
                            WitnessKind::Log,
                            format!("{} > 0", var_name),
                            format!("from ln({})", var_name),
                        )
                    }
                    ImplicitCondition::LowerBound(t, lower) => {
                        let var_name = format_expr_short(ctx, *t);
                        (
                            *t,
                            WitnessKind::Sqrt,
                            format!("{} >= {}", var_name, lower),
                            format!("from lower bound on {}", var_name),
                        )
                    }
                    ImplicitCondition::NonZero(_) => continue,
                };

                let covered_by_stronger =
                    is_covered_by_stronger_predicate(ctx, &cond, root, rewritten_node, replacement);

                if covered_by_stronger
                    || witness_survives_in_context(
                        ctx,
                        target,
                        root,
                        rewritten_node,
                        Some(replacement),
                        kind,
                    )
                {
                    continue;
                }

                unsatisfied.push(cond);
                sources.push(format!("{} ({})", predicate_str, source_str));
            }

            if unsatisfied.is_empty() {
                AnalyticExpansionResult::Safe
            } else {
                AnalyticExpansionResult::WouldExpand {
                    dropped: unsatisfied,
                    sources,
                }
            }
        }
    }
}

/// Quick check: does this rewrite expand analytic domain in tree-context mode?
pub fn expands_analytic_in_context<F>(
    ctx: &Context,
    root: ExprId,
    rewritten_node: ExprId,
    replacement: ExprId,
    infer_domain: F,
) -> bool
where
    F: FnMut(&Context, ExprId) -> ImplicitDomain,
{
    check_analytic_expansion(ctx, root, rewritten_node, replacement, infer_domain).would_expand()
}

/// Infer implicit domain constraints from expression structure.
///
/// `real_only` should be true only when the runtime value-domain is Real.
pub fn infer_implicit_domain(ctx: &Context, root: ExprId, real_only: bool) -> ImplicitDomain {
    if !real_only {
        return ImplicitDomain::empty();
    }

    if !contains_variable(ctx, root) {
        return ImplicitDomain::empty();
    }

    let mut domain = ImplicitDomain::default();
    infer_recursive(ctx, root, &mut domain);
    domain
}

/// Derive additional required conditions from equation equality.
///
/// Runtime crates provide the positivity prover callback so this module remains
/// independent from outer helper stacks.
pub fn derive_requires_from_equation<F>(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    _existing: &ImplicitDomain,
    real_only: bool,
    mut prove_positive: F,
) -> Vec<ImplicitCondition>
where
    F: FnMut(&Context, ExprId) -> Proof,
{
    if !real_only {
        return vec![];
    }

    let mut derived = Vec::new();
    let is_abs = |ctx: &Context, e: ExprId| -> bool { extract_abs_argument_view(ctx, e).is_some() };

    let lhs_positive = prove_positive(ctx, lhs);
    if matches!(lhs_positive, Proof::Proven | Proof::ProvenImplicit)
        && has_positivity_structure(ctx, rhs)
    {
        let rhs_check = prove_positive(ctx, rhs);
        if rhs_check != Proof::Disproven && !is_abs(ctx, rhs) {
            add_positive_and_propagate(ctx, rhs, &mut derived, &mut prove_positive);
        }
    }

    let rhs_positive = prove_positive(ctx, rhs);
    if matches!(rhs_positive, Proof::Proven | Proof::ProvenImplicit)
        && has_positivity_structure(ctx, lhs)
    {
        let lhs_check = prove_positive(ctx, lhs);
        if lhs_check != Proof::Disproven && !is_abs(ctx, lhs) {
            add_positive_and_propagate(ctx, lhs, &mut derived, &mut prove_positive);
        }
    }

    derived
}

/// True when `e` is a NON-NEGATIVE even-root form, i.e. its range is `[0, ∞)`:
/// a bare even root (`sqrt(·)`, or `Pow(·, p/q)` with `q` even and `p > 0` — so
/// `x^(1/2)`, `x^(1/4)`, `x^(3/4)`, `x^(5/2)`; odd roots and `1/√` do not match),
/// OR a POSITIVE rational multiple/quotient of one (`3·√x`, `x^(1/4)/2`). A
/// NEGATIVE coefficient (`-√x`, `-2·√x`) is NOT matched — its range is `(-∞, 0]`,
/// needing the OTHER side `≤ 0` (`NonNegative(-other)`), which cannot be built
/// from `&Context`; that sign-folded form stays an honest residual.
fn is_nonnegative_even_root_form(ctx: &Context, e: ExprId) -> bool {
    if extract_sqrt_argument_view(ctx, e).is_some() {
        return true;
    }
    match ctx.get(e) {
        Expr::Pow(_, exp) => as_rational_const(ctx, *exp)
            .is_some_and(|n| is_even_root_exponent(&n) && n.is_positive()),
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if let Some(c) = as_rational_const(ctx, l) {
                return c.is_positive() && is_nonnegative_even_root_form(ctx, r);
            }
            if let Some(c) = as_rational_const(ctx, r) {
                return c.is_positive() && is_nonnegative_even_root_form(ctx, l);
            }
            false
        }
        Expr::Div(num, den) => as_rational_const(ctx, *den)
            .is_some_and(|c| c.is_positive() && is_nonnegative_even_root_form(ctx, *num)),
        _ => false,
    }
}

/// Even-root RANGE condition for an EQUALITY `even_root(g) = R`. A non-negative
/// even-root form has range `[0, ∞)`, so the equation has a real solution only
/// when the OTHER side is `≥ 0`. Returns `NonNegative(other_side)` when EXACTLY
/// ONE side is a non-negative even-root form (see [`is_nonnegative_even_root_form`]
/// — a bare even root OR a positive multiple of one, `3·√x`, `4·x^(1/4)`), and
/// empty otherwise (incl. both sides even roots, each already `≥ 0`).
///
/// THE CALLER MUST GATE THIS TO `op == RelOp::Eq`: for an inequality
/// (`sqrt(x) > a`) the other side may be negative, so the condition is unsound.
/// This is the equation RANGE condition; the radicand DOMAIN (`x ≥ 0`) is added
/// separately by [`infer_implicit_domain`]. Without it `solve(sqrt(x)=a) → {a²}`
/// silently drops the `a ≥ 0` caveat (and a numeric `a < 0` is, redundantly with
/// the existing numeric path, correctly collapsed to "No solution").
pub(crate) fn even_root_range_conditions(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Vec<ImplicitCondition> {
    let lhs_root = is_nonnegative_even_root_form(ctx, lhs);
    let rhs_root = is_nonnegative_even_root_form(ctx, rhs);
    if lhs_root && !rhs_root {
        vec![ImplicitCondition::NonNegative(rhs)]
    } else if rhs_root && !lhs_root {
        vec![ImplicitCondition::NonNegative(lhs)]
    } else {
        vec![]
    }
}

fn is_covered_by_stronger_predicate(
    ctx: &Context,
    predicate: &ImplicitCondition,
    root: ExprId,
    rewritten_node: ExprId,
    replacement: ExprId,
) -> bool {
    match predicate {
        ImplicitCondition::NonNegative(t) => witness_survives_in_context(
            ctx,
            *t,
            root,
            rewritten_node,
            Some(replacement),
            WitnessKind::Log,
        ),
        ImplicitCondition::LowerBound(_, _) => false,
        ImplicitCondition::Positive(_) => false,
        ImplicitCondition::NonZero(_) => false,
        ImplicitCondition::PrincipalBranch { .. } => false,
    }
}

/// Format expression for compact diagnostics.
pub(crate) fn format_expr_short(ctx: &Context, expr: ExprId) -> String {
    match ctx.get(expr) {
        Expr::Variable(sym_id) => ctx.sym_name(*sym_id).to_string(),
        Expr::Number(n) => format!("{}", n),
        _ => format!("expr#{:?}", expr),
    }
}

fn add_positive_and_propagate<F>(
    ctx: &Context,
    expr: ExprId,
    derived: &mut Vec<ImplicitCondition>,
    prove_positive: &mut F,
) where
    F: FnMut(&Context, ExprId) -> Proof,
{
    let positive_check = prove_positive(ctx, expr);
    if positive_check == Proof::Disproven {
        return;
    }

    match ctx.get(expr) {
        Expr::Function(_, _) if extract_abs_argument_view(ctx, expr).is_some() => {
            let Some(arg) = extract_abs_argument_view(ctx, expr) else {
                return;
            };
            derived.push(ImplicitCondition::NonZero(arg));
        }
        Expr::Function(_, _) if extract_sqrt_argument_view(ctx, expr).is_some() => {
            let Some(arg) = extract_sqrt_argument_view(ctx, expr) else {
                return;
            };
            derived.push(ImplicitCondition::Positive(expr));
            derived.push(ImplicitCondition::Positive(arg));
        }
        Expr::Pow(base, exp) => {
            if let Some(n) = as_rational_const(ctx, *exp) {
                if is_even_root_exponent(&n) {
                    derived.push(ImplicitCondition::Positive(expr));
                    derived.push(ImplicitCondition::Positive(*base));
                    return;
                }
            }
            derived.push(ImplicitCondition::Positive(expr));
        }
        _ => {
            derived.push(ImplicitCondition::Positive(expr));
        }
    }
}

fn real_log_base_constant_is_invalid(ctx: &Context, base: ExprId) -> bool {
    if base == log10_base_sentinel() {
        return false;
    }

    as_rational_const(ctx, base).is_some_and(|value| !value.is_positive() || value.is_one())
}

fn sqrt_like_radicand_for_domain(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Some(radicand) = extract_sqrt_argument_view(ctx, expr) {
        return Some(radicand);
    }

    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    let exponent = as_rational_const(ctx, *exponent)?;
    (exponent == BigRational::new(1.into(), 2.into())).then_some(*base)
}

fn infer_recursive(ctx: &Context, root: ExprId, domain: &mut ImplicitDomain) {
    let mut stack = vec![(root, false)];

    while let Some((expr, inside_calculus_call)) = stack.pop() {
        match ctx.get(expr) {
            // `root_sum(R(t), t, body)` BINDS `t`: any condition read off its
            // body would leak the bound variable to the public boundary as a
            // nonsense pointwise predicate (`x - 7·t != 0`). The node is
            // domain-opaque — its real-domain honesty is carried by the
            // integrand's own pole conditions — so do not descend at all.
            Expr::Function(fn_id, _) if ctx.sym_name(*fn_id) == "root_sum" => {}
            Expr::Function(fn_id, args)
                if !inside_calculus_call
                    && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Acosh)
                    && args.len() == 1 =>
            {
                if !matches!(ctx.get(args[0]), Expr::Number(_)) {
                    let bounded = sqrt_like_radicand_for_domain(ctx, args[0]).unwrap_or(args[0]);
                    domain.add_lower_bound(bounded, BigRational::from_integer(1.into()));
                }
                stack.push((args[0], inside_calculus_call));
            }
            Expr::Function(_, _) if extract_sqrt_argument_view(ctx, expr).is_some() => {
                let Some(arg) = extract_sqrt_argument_view(ctx, expr) else {
                    continue;
                };
                if !matches!(ctx.get(arg), Expr::Number(_)) {
                    domain.add_nonnegative(arg);
                }
                stack.push((arg, inside_calculus_call));
            }
            Expr::Function(_, _) if extract_unary_log_argument_view(ctx, expr).is_some() => {
                let Some(arg) = extract_unary_log_argument_view(ctx, expr) else {
                    continue;
                };
                domain.add_positive(arg);
                stack.push((arg, inside_calculus_call));
            }
            Expr::Function(_, _) if extract_log_base_argument_view(ctx, expr).is_some() => {
                let Some((base_opt, arg)) = extract_log_base_argument_view(ctx, expr) else {
                    continue;
                };
                if base_opt.is_some_and(|base| real_log_base_constant_is_invalid(ctx, base)) {
                    continue;
                }

                domain.add_positive(arg);
                stack.push((arg, inside_calculus_call));

                if let Some(base) = base_opt {
                    if base != log10_base_sentinel() && !matches!(ctx.get(base), Expr::Number(_)) {
                        domain.add_positive(base);
                        stack.push((base, inside_calculus_call));
                    }
                }
            }
            Expr::Pow(base, exp) => {
                if let Some(n) = as_rational_const(ctx, *exp) {
                    let base_is_number = matches!(ctx.get(*base), Expr::Number(_));
                    let even_root = is_even_root_exponent(&n);
                    if n.is_negative() && !even_root && !base_is_number {
                        domain.add_nonzero(*base);
                        add_denominator_nonzero_implications(ctx, *base, domain);
                    }
                    if even_root && !base_is_number {
                        if n.is_negative() {
                            domain.add_positive(*base);
                        } else {
                            domain.add_nonnegative(*base);
                        }
                    }
                    // x^0 is defined only for x != 0 (0^0 is undefined), so the
                    // input itself carries the condition even before any rewrite.
                    if n.is_zero() && !base_is_number {
                        domain.add_nonzero(*base);
                    }
                }
                stack.push((*base, inside_calculus_call));
                stack.push((*exp, inside_calculus_call));
            }
            Expr::Div(num, den) => {
                domain.add_nonzero(*den);
                add_denominator_nonzero_implications(ctx, *den, domain);
                stack.push((*num, inside_calculus_call));
                stack.push((*den, inside_calculus_call));
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                stack.push((*l, inside_calculus_call));
                stack.push((*r, inside_calculus_call));
            }
            Expr::Neg(inner) => {
                stack.push((*inner, inside_calculus_call));
            }
            Expr::Function(fn_id, args) => {
                let enters_calculus =
                    matches!(ctx.sym_name(*fn_id), "diff" | "integrate" | "limit");
                stack.extend(
                    args.iter()
                        .copied()
                        .map(|arg| (arg, inside_calculus_call || enters_calculus)),
                );
            }
            Expr::Matrix { data, .. } => {
                stack.extend(data.iter().copied().map(|arg| (arg, inside_calculus_call)));
            }
            Expr::Hold(inner) => {
                stack.push((*inner, inside_calculus_call));
            }
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }
}

fn add_denominator_nonzero_implications(
    ctx: &Context,
    denominator: ExprId,
    domain: &mut ImplicitDomain,
) {
    let mut current = denominator;

    loop {
        let Expr::Pow(base, exp) = ctx.get(current) else {
            return;
        };
        let Expr::Number(n) = ctx.get(*exp) else {
            return;
        };
        if !n.is_integer() || *n <= num_rational::BigRational::zero() {
            return;
        }
        if !matches!(ctx.get(*base), Expr::Number(_)) {
            domain.add_nonzero(*base);
        }
        current = *base;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        derive_requires_from_equation, domain_delta_check, infer_implicit_domain, DomainDelta,
    };
    use crate::domain_condition::{ImplicitCondition, ImplicitDomain};
    use crate::domain_proof::Proof;
    use cas_ast::{Context, Expr};

    #[test]
    fn infer_respects_real_only_flag() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);

        let real_domain = infer_implicit_domain(&ctx, sqrt_x, true);
        let complex_domain = infer_implicit_domain(&ctx, sqrt_x, false);

        assert!(real_domain.contains_nonnegative(x));
        assert!(complex_domain.is_empty());
    }

    #[test]
    fn delta_detects_dropped_analytic_condition() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);
        let two = ctx.num(2);
        let input = ctx.add(Expr::Pow(sqrt_x, two));
        let output = x;

        let delta = domain_delta_check(&ctx, input, output, |ctx, expr| {
            infer_implicit_domain(ctx, expr, true)
        });

        assert!(matches!(delta, DomainDelta::ExpandsAnalytic(_)));
    }

    #[test]
    fn derive_positive_through_sqrt() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let lhs = ctx.num(2);
        let rhs = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);

        let derived = derive_requires_from_equation(
            &ctx,
            lhs,
            rhs,
            &ImplicitDomain::empty(),
            true,
            |_, expr| {
                if expr == lhs {
                    Proof::Proven
                } else {
                    Proof::Unknown
                }
            },
        );

        assert!(derived.contains(&ImplicitCondition::Positive(rhs)));
        assert!(derived.contains(&ImplicitCondition::Positive(x)));
    }

    #[test]
    fn infer_division_by_positive_integer_power_adds_base_nonzero() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let expr = ctx.add(Expr::Div(one, x_squared));

        let domain = infer_implicit_domain(&ctx, expr, true);

        assert!(domain.contains_nonzero(x_squared));
        assert!(domain.contains_nonzero(x));
    }

    #[test]
    fn infer_negative_even_root_power_adds_positive_base() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let minus_one = ctx.num(-1);
        let two = ctx.num(2);
        let exponent = ctx.add(Expr::Div(minus_one, two));
        let expr = ctx.add(Expr::Pow(x, exponent));

        let domain = infer_implicit_domain(&ctx, expr, true);

        assert!(domain.contains_positive(x));
        assert!(!domain.contains_nonnegative(x));
        assert!(!domain.contains_nonzero(x));
    }

    #[test]
    fn even_root_range_conditions_constrain_the_non_radical_side() {
        use super::even_root_range_conditions;
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let a = ctx.var("a");
        let nn_a = ImplicitCondition::NonNegative(a);

        // sqrt(x) = a  =>  a >= 0 (even root range); symmetric when the root is RHS.
        let sqrt_x = ctx.call("sqrt", vec![x]);
        assert_eq!(
            even_root_range_conditions(&ctx, sqrt_x, a),
            vec![nn_a.clone()]
        );
        assert_eq!(
            even_root_range_conditions(&ctx, a, sqrt_x),
            vec![nn_a.clone()]
        );

        // x^(1/4) = a (even denominator, positive numerator) => a >= 0.
        let one = ctx.num(1);
        let four = ctx.num(4);
        let quarter = ctx.add(Expr::Div(one, four));
        let x_quart = ctx.add(Expr::Pow(x, quarter));
        assert_eq!(
            even_root_range_conditions(&ctx, x_quart, a),
            vec![nn_a.clone()]
        );

        // ODD root x^(1/3) = a: no range condition.
        let one3 = ctx.num(1);
        let three = ctx.num(3);
        let third = ctx.add(Expr::Div(one3, three));
        let x_cube = ctx.add(Expr::Pow(x, third));
        assert!(even_root_range_conditions(&ctx, x_cube, a).is_empty());

        // POSITIVE coefficient on the radical (3·√x = a) still constrains a ≥ 0.
        let three_c = ctx.num(3);
        let three_sqrt_x = ctx.add(Expr::Mul(three_c, sqrt_x));
        assert_eq!(
            even_root_range_conditions(&ctx, three_sqrt_x, a),
            vec![nn_a.clone()]
        );
        // √x / 2 = a likewise.
        let two_d = ctx.num(2);
        let sqrt_x_half = ctx.add(Expr::Div(sqrt_x, two_d));
        assert_eq!(
            even_root_range_conditions(&ctx, sqrt_x_half, a),
            vec![nn_a.clone()]
        );
        // NEGATIVE coefficient (-√x): range is (-∞,0], needing a ≤ 0 which cannot be
        // built here — emit NOTHING (a sound under-answer, never a WRONG a ≥ 0).
        let neg_one_c = ctx.num(-1);
        let neg_sqrt_x = ctx.add(Expr::Mul(neg_one_c, sqrt_x));
        assert!(even_root_range_conditions(&ctx, neg_sqrt_x, a).is_empty());

        // BOTH sides even roots: each already >= 0, no extra condition.
        let sqrt_a = ctx.call("sqrt", vec![a]);
        assert!(even_root_range_conditions(&ctx, sqrt_x, sqrt_a).is_empty());

        // NEGATIVE even exponent (1/sqrt = x^(-1/2)): NOT a [0,∞) range form here
        // (the radicand/positivity path owns it) — no range condition emitted.
        let neg_one = ctx.num(-1);
        let two = ctx.num(2);
        let neg_half = ctx.add(Expr::Div(neg_one, two));
        let recip_sqrt = ctx.add(Expr::Pow(x, neg_half));
        assert!(even_root_range_conditions(&ctx, recip_sqrt, a).is_empty());
    }

    #[test]
    fn infer_negative_integer_power_adds_base_nonzero() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let one = ctx.num(1);
        let denominator = ctx.add(Expr::Add(y, one));
        let minus_one = ctx.num(-1);
        let expr = ctx.add(Expr::Pow(denominator, minus_one));

        let domain = infer_implicit_domain(&ctx, expr, true);

        assert!(domain.contains_nonzero(denominator));
    }

    #[test]
    fn infer_general_log_adds_positive_base_and_argument() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let x = ctx.var("x");
        let expr = ctx.call_builtin(cas_ast::BuiltinFn::Log, vec![b, x]);

        let domain = infer_implicit_domain(&ctx, expr, true);

        assert!(domain.contains_positive(b));
        assert!(domain.contains_positive(x));
    }

    #[test]
    fn infer_general_log_invalid_constant_base_skips_argument_condition() {
        for base_value in [-2, 0, 1] {
            let mut ctx = Context::new();
            let base = ctx.num(base_value);
            let x = ctx.var("x");
            let expr = ctx.call_builtin(cas_ast::BuiltinFn::Log, vec![base, x]);

            let domain = infer_implicit_domain(&ctx, expr, true);

            assert!(
                !domain.contains_positive(x),
                "base {base_value} should make the log invalid before argument conditions"
            );
        }
    }
}
