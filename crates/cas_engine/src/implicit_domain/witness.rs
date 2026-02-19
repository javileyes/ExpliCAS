//! Witness survival checks and assumption classification.

use super::inference::is_even_root_exponent;
use super::normalization::{
    conditions_equivalent, exprs_equivalent, is_odd_power_of, is_positive_multiple_of,
    is_power_of_base,
};
use super::ImplicitCondition;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_extract::{extract_sqrt_argument_view, extract_unary_log_argument_view};

// =============================================================================
// Witness Survival
// =============================================================================

/// Kind of witness to look for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WitnessKind {
    /// sqrt(t) or t^(1/2) for NonNegative(t)
    Sqrt,
    /// ln(t) or log(t) for Positive(t)
    Log,
    /// 1/t or Div(_, t) for NonZero(t)
    Division,
}

/// Check if a witness for a condition survives in the output expression.
///
/// This is the critical safety check: a condition from implicit domain
/// is only valid if the witness that enforces it still exists in the output.
///
/// # Arguments
/// * `ctx` - AST context
/// * `target` - The target expression (e.g., `x` in `x ≥ 0`)
/// * `output` - The expression to search for witnesses
/// * `kind` - What kind of witness to look for
///
/// # Returns
/// `true` if a witness survives in output, `false` otherwise.
pub fn witness_survives(ctx: &Context, target: ExprId, output: ExprId, kind: WitnessKind) -> bool {
    search_witness(ctx, target, output, kind)
}

/// Iterative witness search to prevent stack overflow on deep expressions.
pub(crate) fn search_witness(
    ctx: &Context,
    target: ExprId,
    root: ExprId,
    kind: WitnessKind,
) -> bool {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            // Check if this node is a witness: sqrt(target)
            Expr::Function(_, _) if extract_sqrt_argument_view(ctx, expr).is_some() => {
                let Some(arg) = extract_sqrt_argument_view(ctx, expr) else {
                    continue;
                };
                if kind == WitnessKind::Sqrt && exprs_equal(ctx, arg, target) {
                    return true;
                }
                stack.push(arg);
            }

            // Check if this node is a witness: ln(target) or log(target)
            Expr::Function(_, _) if extract_unary_log_argument_view(ctx, expr).is_some() => {
                let Some(arg) = extract_unary_log_argument_view(ctx, expr) else {
                    continue;
                };
                if kind == WitnessKind::Log && exprs_equal(ctx, arg, target) {
                    return true;
                }
                stack.push(arg);
            }

            // Check for t^(1/2) form as witness for sqrt
            Expr::Pow(base, exp) => {
                if kind == WitnessKind::Sqrt {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        if is_even_root_exponent(n) && exprs_equal(ctx, *base, target) {
                            return true;
                        }
                    }
                }
                stack.push(*base);
                stack.push(*exp);
            }

            // Check for Div(_, target) as witness for division
            Expr::Div(num, den) => {
                if kind == WitnessKind::Division && exprs_equal(ctx, *den, target) {
                    return true;
                }
                stack.push(*num);
                stack.push(*den);
            }

            // Push children for traversal
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),

            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

/// Check if two expressions are equal (by ExprId or structural comparison).
pub(crate) fn exprs_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    // Use ordering comparison for structural equality
    crate::ordering::compare_expr(ctx, a, b) == std::cmp::Ordering::Equal
}

/// Check if a witness survives in the context of the full expression tree,
/// considering that a specific node is being replaced with a new value.
///
/// This is the key function for implicit domain safety: it ensures that when
/// we simplify `sqrt(x)² → x`, a witness (`sqrt(x)`) still exists elsewhere
/// in the expression tree.
///
/// # Arguments
/// * `ctx` - AST context
/// * `target` - The target expression (e.g., `x` in `x ≥ 0`)
/// * `root` - The root expression of the full tree
/// * `replaced_node` - The node being replaced (will be skipped in search)
/// * `replacement` - Optional replacement value to search in instead
/// * `kind` - What kind of witness to look for
///
/// # Returns
/// `true` if a witness survives in the tree after replacement
pub fn witness_survives_in_context(
    ctx: &Context,
    target: ExprId,
    root: ExprId,
    replaced_node: ExprId,
    replacement: Option<ExprId>,
    kind: WitnessKind,
) -> bool {
    search_witness_in_context(ctx, target, root, replaced_node, replacement, kind)
}

/// Iterative witness search with node replacement to prevent stack overflow.
fn search_witness_in_context(
    ctx: &Context,
    target: ExprId,
    root: ExprId,
    replaced_node: ExprId,
    replacement: Option<ExprId>,
    kind: WitnessKind,
) -> bool {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        // If we've reached the replaced node, search in replacement instead
        if expr == replaced_node {
            if let Some(repl) = replacement {
                if search_witness(ctx, target, repl, kind) {
                    return true;
                }
            }
            // Skip children of replaced node
            continue;
        }

        match ctx.get(expr) {
            // Check if this node is a witness: sqrt(target)
            Expr::Function(_, _) if extract_sqrt_argument_view(ctx, expr).is_some() => {
                let Some(arg) = extract_sqrt_argument_view(ctx, expr) else {
                    continue;
                };
                if kind == WitnessKind::Sqrt && exprs_equal(ctx, arg, target) {
                    return true;
                }
                stack.push(arg);
            }

            // Check if this node is a witness: ln(target) or log(target)
            Expr::Function(_, _) if extract_unary_log_argument_view(ctx, expr).is_some() => {
                let Some(arg) = extract_unary_log_argument_view(ctx, expr) else {
                    continue;
                };
                if kind == WitnessKind::Log && exprs_equal(ctx, arg, target) {
                    return true;
                }
                stack.push(arg);
            }

            // Check for t^(1/2) form as witness for sqrt
            Expr::Pow(base, exp) => {
                if kind == WitnessKind::Sqrt {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        if is_even_root_exponent(n) && exprs_equal(ctx, *base, target) {
                            return true;
                        }
                    }
                }
                stack.push(*base);
                stack.push(*exp);
            }

            // Check for Div(_, target) as witness for division
            Expr::Div(num, den) => {
                if kind == WitnessKind::Division && exprs_equal(ctx, *den, target) {
                    return true;
                }
                stack.push(*num);
                stack.push(*den);
            }

            // Push children for traversal
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),

            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

// =============================================================================
// Assumption Classification (V2.12.13)
// =============================================================================

/// Context for domain condition tracking during step processing.
///
/// Used by the central classifier to determine whether conditions are
/// derived from input requires or newly introduced.
#[derive(Debug, Clone, Default)]
pub struct DomainContext {
    /// Conditions inferred from the original input expression
    pub global_requires: Vec<ImplicitCondition>,
    /// Conditions introduced by previous steps (accumulated)
    pub introduced_requires: Vec<ImplicitCondition>,
}

impl DomainContext {
    /// Create a new DomainContext with global requires from the input expression.
    pub fn new(global_requires: Vec<ImplicitCondition>) -> Self {
        Self {
            global_requires,
            introduced_requires: Vec::new(),
        }
    }

    /// Check if a condition is implied by the known requires (global ∪ introduced).
    ///
    /// Implication rules:
    /// - Exact polynomial equivalence
    /// - x > 0 is implied by x^(odd positive) > 0  
    /// - x ≠ 0 is implied by x > 0
    pub fn is_condition_implied(&self, ctx: &Context, cond: &ImplicitCondition) -> bool {
        let all_known: Vec<_> = self
            .global_requires
            .iter()
            .chain(self.introduced_requires.iter())
            .collect();

        for known in all_known {
            // Direct equivalence check
            if conditions_equivalent(ctx, cond, known) {
                return true;
            }

            // Implication rules
            match (cond, known) {
                // x ≠ 0 is implied by x > 0 or x ≥ 0 (for our purposes, x > 0)
                (ImplicitCondition::NonZero(target), ImplicitCondition::Positive(source)) => {
                    if exprs_equivalent(ctx, *target, *source) {
                        return true;
                    }
                }
                // x > 0 is implied by x^(positive odd) > 0 (e.g., b > 0 implied by b^3 > 0)
                // x^n > 0 is implied by x > 0 (e.g., a^2 > 0 implied by a > 0)
                (ImplicitCondition::Positive(target), ImplicitCondition::Positive(source)) => {
                    // Check if source is target^(odd positive) -> target is implied
                    if is_odd_power_of(ctx, *source, *target) {
                        return true;
                    }
                    // NEW: Check if target is source^n -> target is implied by source
                    // e.g., a^2 > 0 is implied by a > 0
                    if is_power_of_base(ctx, *target, *source) {
                        return true;
                    }
                }
                // V2.15.8: x ≥ 0 is implied by x > 0 (strict positivity implies non-negativity)
                (ImplicitCondition::NonNegative(target), ImplicitCondition::Positive(source)) => {
                    if exprs_equivalent(ctx, *target, *source) {
                        return true;
                    }
                }
                // V2.15.8: x ≥ 0 is implied by k*x ≥ 0 when k > 0
                // e.g., x ≥ 0 implied by 4*x ≥ 0 (since we know 4 > 0)
                (
                    ImplicitCondition::NonNegative(target),
                    ImplicitCondition::NonNegative(source),
                ) => {
                    // Check direct equivalence first (handled above in conditions_equivalent)
                    // Check if source is k*target where k > 0
                    if is_positive_multiple_of(ctx, *source, *target) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Add a newly introduced condition (from a step that introduces constraints).
    pub fn add_introduced(&mut self, cond: ImplicitCondition) {
        self.introduced_requires.push(cond);
    }
}

/// Classify an AssumptionEvent based on whether its condition is implied by known requires.
///
/// # Reclassification Logic (V2.12.13)
///
/// 1. If the event has kind `BranchChoice`, `HeuristicAssumption`, or `DomainExtension`:
///    **Keep as-is** (never promote to requires)
///
/// 2. If the event's condition IS implied by `global ∪ introduced`:
///    Reclassify to `DerivedFromRequires` (will not be displayed)
///
/// 3. If NOT implied AND kind was `DerivedFromRequires` or `RequiresIntroduced`:
///    Promote to `RequiresIntroduced` (will be displayed, add to introduced)
///
/// # Returns
/// The new kind for the event, and whether to add to introduced_requires.
pub fn classify_assumption(
    ctx: &Context,
    dc: &DomainContext,
    event: &crate::assumptions::AssumptionEvent,
) -> (
    crate::assumptions::AssumptionKind,
    Option<ImplicitCondition>,
) {
    use crate::assumptions::AssumptionKind;

    // Rule 1: Branch/Domain never get reclassified (they are structural, not algebraic)
    match event.kind {
        AssumptionKind::BranchChoice | AssumptionKind::DomainExtension => {
            return (event.kind, None);
        }
        _ => {}
    }

    // Try to convert the event to an ImplicitCondition
    let implicit_cond = assumption_to_condition(event);

    match implicit_cond {
        Some(cond) => {
            // Check if this condition is already implied by global/introduced requires
            if dc.is_condition_implied(ctx, &cond) {
                // If it was HeuristicAssumption but is implied, downgrade to DerivedFromRequires
                // This prevents showing ⚠ b > 0 when b > 0 is already in Requires
                (AssumptionKind::DerivedFromRequires, None)
            } else {
                // Not implied - behavior depends on original kind
                match event.kind {
                    AssumptionKind::HeuristicAssumption => {
                        // Keep as HeuristicAssumption (shows ⚠) since it's a new assumption
                        (AssumptionKind::HeuristicAssumption, None)
                    }
                    _ => {
                        // Promote to RequiresIntroduced
                        (AssumptionKind::RequiresIntroduced, Some(cond))
                    }
                }
            }
        }
        None => {
            // Cannot convert to condition (e.g., InvTrigPrincipalRange)
            // Keep original kind
            (event.kind, None)
        }
    }
}

/// Convert an AssumptionEvent to an ImplicitCondition if possible.
///
/// Uses the `expr_id` field (V2.12.13) for proper condition comparison.
fn assumption_to_condition(
    event: &crate::assumptions::AssumptionEvent,
) -> Option<ImplicitCondition> {
    use crate::assumptions::AssumptionKey;

    // V2.12.13: Use expr_id if available for proper condition creation
    let expr_id = event.expr_id?;

    match &event.key {
        AssumptionKey::NonZero { .. } => Some(ImplicitCondition::NonZero(expr_id)),
        AssumptionKey::Positive { .. } => Some(ImplicitCondition::Positive(expr_id)),
        AssumptionKey::NonNegative { .. } => Some(ImplicitCondition::NonNegative(expr_id)),
        // Defined has no direct ImplicitCondition counterpart
        AssumptionKey::Defined { .. } => None,
        // Branch choices are not conditions
        AssumptionKey::InvTrigPrincipalRange { .. } => None,
        AssumptionKey::ComplexPrincipalBranch { .. } => None,
    }
}

/// Filter and reclassify a list of AssumptionEvents in place.
///
/// After calling, events have updated `kind` fields.
/// Use `event.kind.should_display()` to determine which to show.
pub fn classify_assumptions_in_place(
    ctx: &Context,
    dc: &mut DomainContext,
    events: &mut [crate::assumptions::AssumptionEvent],
) {
    for event in events.iter_mut() {
        let (new_kind, new_cond) = classify_assumption(ctx, dc, event);
        event.kind = new_kind;
        if let Some(cond) = new_cond {
            dc.add_introduced(cond);
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::semantics::ValueDomain;
    use cas_ast::Context;

    #[test]
    fn test_infer_sqrt_implies_nonnegative() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);

        let domain = infer_implicit_domain(&ctx, sqrt_x, ValueDomain::RealOnly);

        assert!(domain.contains_nonnegative(x));
        assert!(!domain.contains_positive(x));
    }

    #[test]
    fn test_infer_ln_implies_positive() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let ln_x = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![x]);

        let domain = infer_implicit_domain(&ctx, ln_x, ValueDomain::RealOnly);

        assert!(domain.contains_positive(x));
    }

    #[test]
    fn test_infer_div_implies_nonzero() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let one_over_x = ctx.add(Expr::Div(one, x));

        let domain = infer_implicit_domain(&ctx, one_over_x, ValueDomain::RealOnly);

        assert!(domain.contains_nonzero(x));
    }

    #[test]
    fn test_witness_survives_sqrt() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);
        let y = ctx.var("y");
        let output = ctx.add(Expr::Add(sqrt_x, y)); // sqrt(x) + y

        assert!(witness_survives(&ctx, x, output, WitnessKind::Sqrt));
    }

    #[test]
    fn test_witness_not_survives() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        // Output is just x, no sqrt(x) witness

        assert!(!witness_survives(&ctx, x, x, WitnessKind::Sqrt));
    }

    #[test]
    fn test_complex_enabled_returns_empty() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);

        let domain = infer_implicit_domain(&ctx, sqrt_x, ValueDomain::ComplexEnabled);

        assert!(domain.is_empty());
    }

    #[test]
    fn test_domain_delta_sqrt_square_to_x() {
        // sqrt(x)^2 -> x should be detected as ExpandsAnalytic
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);
        let two = ctx.num(2);
        let sqrt_x_squared = ctx.add(Expr::Pow(sqrt_x, two));

        // Check input has NonNegative(x)
        let d_in = infer_implicit_domain(&ctx, sqrt_x_squared, ValueDomain::RealOnly);
        assert!(
            d_in.contains_nonnegative(x),
            "Input should have NonNegative(x)"
        );

        // Check output (x) has no NonNegative
        let d_out = infer_implicit_domain(&ctx, x, ValueDomain::RealOnly);
        assert!(
            d_out.is_empty(),
            "Output (just x) should have no constraints"
        );

        // Check domain_delta_check detects this as ExpandsAnalytic
        let delta = domain_delta_check(&ctx, sqrt_x_squared, x, ValueDomain::RealOnly);
        assert!(
            matches!(delta, DomainDelta::ExpandsAnalytic(_)),
            "sqrt(x)^2 -> x should be detected as ExpandsAnalytic, got {:?}",
            delta
        );
    }

    #[test]
    fn test_domain_delta_safe_with_witness_preserved() {
        // (x-y)/(sqrt(x)-sqrt(y)) -> sqrt(x)+sqrt(y) preserves sqrt witnesses
        // This is a simplified version - we just test that sqrt in output means safe
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);
        let y = ctx.var("y");
        let sqrt_y = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![y]);

        // Input: sqrt(x) - sqrt(y)
        let input = ctx.add(Expr::Sub(sqrt_x, sqrt_y));
        // Output: sqrt(x) + sqrt(y)
        let output = ctx.add(Expr::Add(sqrt_x, sqrt_y));

        let delta = domain_delta_check(&ctx, input, output, ValueDomain::RealOnly);
        assert_eq!(
            delta,
            DomainDelta::Safe,
            "sqrt witnesses preserved should be safe"
        );
    }

    #[test]
    fn test_search_witness_deep_expression_no_overflow() {
        // Regression test: a 500-deep nested expression would overflow
        // the stack with the old recursive implementation.
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);

        // Build: Add(Add(Add(... sqrt(x) ..., 1), 1), 1) — depth 500
        let mut expr = sqrt_x;
        for _ in 0..500 {
            let one = ctx.num(1);
            expr = ctx.add(Expr::Add(expr, one));
        }

        assert!(
            witness_survives(&ctx, x, expr, WitnessKind::Sqrt),
            "sqrt(x) witness should survive deep nesting"
        );

        // Also verify the in-context variant doesn't overflow
        let dummy_replaced = ctx.num(999);
        assert!(
            witness_survives_in_context(&ctx, x, expr, dummy_replaced, None, WitnessKind::Sqrt,),
            "in-context search should survive deep nesting"
        );
    }
}
