//! Implicit-domain inference and domain-delta checks.
//!
//! This module is runtime-agnostic. Runtime crates provide:
//! - whether inference is enabled (`real_only`)
//! - positivity prover callback
//! - domain inference callback when comparing rewrites

use crate::domain_condition::{ImplicitCondition, ImplicitDomain};
use crate::domain_proof::Proof;
use crate::domain_witness::{witness_survives_in_context, WitnessKind};
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_extract::{
    extract_abs_argument_view, extract_sqrt_argument_view, extract_unary_log_argument_view,
};
use cas_math::expr_predicates::{
    contains_variable, has_positivity_structure, is_even_root_exponent,
};

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
                ImplicitCondition::NonNegative(_) | ImplicitCondition::Positive(_)
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
        ImplicitCondition::Positive(_) => false,
        ImplicitCondition::NonZero(_) => false,
    }
}

/// Format expression for compact diagnostics.
pub fn format_expr_short(ctx: &Context, expr: ExprId) -> String {
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
            if let Expr::Number(n) = ctx.get(*exp) {
                if is_even_root_exponent(n) {
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

fn infer_recursive(ctx: &Context, root: ExprId, domain: &mut ImplicitDomain) {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(_, _) if extract_sqrt_argument_view(ctx, expr).is_some() => {
                let Some(arg) = extract_sqrt_argument_view(ctx, expr) else {
                    continue;
                };
                if !matches!(ctx.get(arg), Expr::Number(_)) {
                    domain.add_nonnegative(arg);
                }
                stack.push(arg);
            }
            Expr::Function(_, _) if extract_unary_log_argument_view(ctx, expr).is_some() => {
                let Some(arg) = extract_unary_log_argument_view(ctx, expr) else {
                    continue;
                };
                domain.add_positive(arg);
                stack.push(arg);
            }
            Expr::Pow(base, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if is_even_root_exponent(n) && !matches!(ctx.get(*base), Expr::Number(_)) {
                        domain.add_nonnegative(*base);
                    }
                }
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Div(num, den) => {
                domain.add_nonzero(*den);
                stack.push(*num);
                stack.push(*den);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => {
                stack.push(*inner);
            }
            Expr::Function(_, args) => {
                stack.extend(args.iter().copied());
            }
            Expr::Matrix { data, .. } => {
                stack.extend(data.iter().copied());
            }
            Expr::Hold(inner) => {
                stack.push(*inner);
            }
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
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
}
