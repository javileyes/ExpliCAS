//! Domain delta checks, implicit domain inference, and related helpers.

use super::witness::{witness_survives_in_context, WitnessKind};
use super::{ImplicitCondition, ImplicitDomain};
use crate::semantics::ValueDomain;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_extract::{extract_sqrt_argument_view, extract_unary_log_argument_view};
use num_integer::Integer;
use num_rational::BigRational;

// =============================================================================
// Domain Delta Check (Airbag)
// =============================================================================

/// Result of domain delta check between input and output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomainDelta {
    /// No domain expansion (output has same or more constraints)
    Safe,
    /// Domain was expanded by removing Analytic constraints (x≥0, x>0)
    ExpandsAnalytic(Vec<ImplicitCondition>),
    /// Domain was expanded by removing Definability constraints (x≠0)
    ExpandsDefinability(Vec<ImplicitCondition>),
}

/// Check if a rewrite would expand the domain by removing implicit constraints.
///
/// This is the "airbag" function: it compares the implicit domains of input
/// and output, and detects if the rewrite removes constraints.
///
/// # Arguments
/// * `ctx` - AST context
/// * `input` - Original expression
/// * `output` - Rewritten expression
/// * `vd` - Value domain (RealOnly/ComplexEnabled)
///
/// # Returns
/// * `DomainDelta::Safe` if output has same or more constraints
/// * `DomainDelta::ExpandsAnalytic` if Analytic constraints were removed (x≥0, x>0)
/// * `DomainDelta::ExpandsDefinability` if only Definability constraints were removed (x≠0)
pub fn domain_delta_check(
    ctx: &Context,
    input: ExprId,
    output: ExprId,
    vd: ValueDomain,
) -> DomainDelta {
    let d_in = infer_implicit_domain(ctx, input, vd);
    let d_out = infer_implicit_domain(ctx, output, vd);

    // Find conditions in input that are NOT in output (dropped conditions)
    let dropped: Vec<ImplicitCondition> = d_in
        .conditions()
        .iter()
        .filter(|c| !d_out.conditions().contains(c))
        .cloned()
        .collect();

    if dropped.is_empty() {
        return DomainDelta::Safe;
    }

    // Separate by type: Analytic (NonNegative, Positive) vs Definability (NonZero)
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
/// Use this as a guard in Strict/Generic modes.
pub fn expands_analytic_domain(
    ctx: &Context,
    input: ExprId,
    output: ExprId,
    vd: ValueDomain,
) -> bool {
    matches!(
        domain_delta_check(ctx, input, output, vd),
        DomainDelta::ExpandsAnalytic(_)
    )
}

/// Context-aware check: does this rewrite expand analytic domain considering the full tree?
///
/// This version checks if the witnesses for dropped constraints survive elsewhere in the tree.
/// If they do, the rewrite is safe even if the local transformation removes them.
///
/// # Arguments
/// * `ctx` - AST context
/// * `root` - Root of the full expression tree
/// * `rewritten_node` - The node being replaced
/// * `replacement` - The replacement expression
/// * `vd` - Value domain
///
/// Result of checking if a rewrite would expand analytic domain.
#[derive(Debug, Clone)]
pub enum AnalyticExpansionResult {
    /// No expansion - rewrite is safe
    Safe,
    /// Would expand domain - contains predicates whose witnesses don't survive
    /// Fields: (dropped_predicates, source_descriptions)
    WouldExpand {
        dropped: Vec<ImplicitCondition>,
        sources: Vec<String>, // e.g., "x ≥ 0 (from sqrt(x))"
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

/// Context-aware check: does this rewrite expand analytic domain considering the full tree?
///
/// Returns detailed information about dropped predicates for:
/// - Blocking in Strict/Generic
/// - Registering assumptions in Assume mode
/// - Better UX diagnostics
pub fn check_analytic_expansion(
    ctx: &Context,
    root: ExprId,
    rewritten_node: ExprId,
    replacement: ExprId,
    vd: ValueDomain,
) -> AnalyticExpansionResult {
    // First check if local rewrite expands domain
    let delta = domain_delta_check(ctx, rewritten_node, replacement, vd);

    match delta {
        DomainDelta::Safe => AnalyticExpansionResult::Safe,
        DomainDelta::ExpandsDefinability(_) => AnalyticExpansionResult::Safe, // Only care about Analytic
        DomainDelta::ExpandsAnalytic(dropped) => {
            let mut unsatisfied: Vec<ImplicitCondition> = Vec::new();
            let mut sources: Vec<String> = Vec::new();

            // For each dropped constraint, check if witness survives in tree after replacement
            for cond in dropped {
                let (target, kind, predicate_str, source_str) = match &cond {
                    ImplicitCondition::NonNegative(t) => {
                        let var_name = format_expr_short(ctx, *t);
                        (
                            *t,
                            WitnessKind::Sqrt,
                            format!("{} ≥ 0", var_name),
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
                    ImplicitCondition::NonZero(_) => continue, // Skip definability
                };

                // Check for predicate implication: Positive(x) implies NonNegative(x)
                // If output has ln(x), we have x > 0 which covers x ≥ 0
                let covered_by_stronger =
                    is_covered_by_stronger_predicate(ctx, &cond, root, rewritten_node, replacement);

                // If witness survives OR covered by stronger predicate, it's fine
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
                    continue; // This predicate is satisfied
                }

                // Predicate would be dropped
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

/// Check if a predicate is covered by a stronger predicate in the output.
/// E.g., NonNegative(x) is covered if Positive(x) survives (because x > 0 ⇒ x ≥ 0)
fn is_covered_by_stronger_predicate(
    ctx: &Context,
    predicate: &ImplicitCondition,
    root: ExprId,
    rewritten_node: ExprId,
    replacement: ExprId,
) -> bool {
    match predicate {
        ImplicitCondition::NonNegative(t) => {
            // x ≥ 0 is covered by x > 0 (which comes from ln(x))
            witness_survives_in_context(
                ctx,
                *t,
                root,
                rewritten_node,
                Some(replacement),
                WitnessKind::Log,
            )
        }
        ImplicitCondition::Positive(_) => false, // Nothing stronger than Positive for our purposes
        ImplicitCondition::NonZero(_) => false,  // Not checking definability here
    }
}

/// Format expression for display (short form)
pub(crate) fn format_expr_short(ctx: &Context, expr: ExprId) -> String {
    match ctx.get(expr) {
        Expr::Variable(sym_id) => ctx.sym_name(*sym_id).to_string(),
        Expr::Number(n) => format!("{}", n),
        _ => format!("expr#{:?}", expr),
    }
}

/// Quick check: does this rewrite expand analytic domain?
/// Use this as a simple guard in Strict/Generic modes.
pub fn expands_analytic_in_context(
    ctx: &Context,
    root: ExprId,
    rewritten_node: ExprId,
    replacement: ExprId,
    vd: ValueDomain,
) -> bool {
    check_analytic_expansion(ctx, root, rewritten_node, replacement, vd).would_expand()
}

// =============================================================================
// Inference
// =============================================================================

/// Infer implicit domain constraints from expression structure.
///
/// Only operates in RealOnly mode. Returns empty in ComplexEnabled.
///
/// Traverses the AST and collects:
/// - `sqrt(t)` or `t^(1/2)` → NonNegative(t)
/// - `ln(t)` or `log(t)` → Positive(t)
/// - `1/t` or `Div(_, t)` → NonZero(t)
///
/// NOTE: Skips expressions that don't contain variables - those are fully numeric
/// and don't need implicit domain protection.
pub fn infer_implicit_domain(ctx: &Context, root: ExprId, vd: ValueDomain) -> ImplicitDomain {
    // Track call count for regression testing
    super::infer_domain_calls_inc();

    // Only apply in RealOnly mode
    if vd != ValueDomain::RealOnly {
        return ImplicitDomain::empty();
    }

    // Skip if expression has no variables - fully numeric expressions don't need implicit domain
    if !contains_variable(ctx, root) {
        return ImplicitDomain::empty();
    }

    let mut domain = ImplicitDomain::default();
    infer_recursive(ctx, root, &mut domain);
    domain
}

/// Derive additional required conditions from equation equality.
///
/// This function uses the equation `lhs = rhs` to derive stronger conditions.
/// For RealOnly:
/// - If `prove_positive(lhs)` is proven → add `Positive(rhs)`
/// - If `prove_positive(rhs)` is proven → add `Positive(lhs)`
/// - Then propagate through sqrt: `Positive(sqrt(t))` → `Positive(t)`
///
/// This enables `2^x = sqrt(y)` to derive `y > 0`:
/// - `2^x > 0` (always true for a > 0)
/// - Therefore `sqrt(y) > 0`
/// - Therefore `y > 0`
///
/// Returns additional conditions to add to the required set.
pub fn derive_requires_from_equation(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    _existing: &ImplicitDomain,
    vd: ValueDomain,
) -> Vec<ImplicitCondition> {
    if vd != ValueDomain::RealOnly {
        return vec![];
    }

    let mut derived = Vec::new();

    // Helper to check if expr is abs(...)
    let is_abs = |ctx: &Context, e: ExprId| -> bool {
        matches!(ctx.get(e), Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1)
    };

    // Helper to check if expression has structural constraints that enforce positivity.
    // Only these expressions benefit from derived positivity requirements.
    // Examples: sqrt(y), ln(y), y^(1/2), etc.
    // We DON'T want to derive positivity for plain polynomials like "2*x + 3".
    let has_positivity_structure = |ctx: &Context, e: ExprId| -> bool {
        if extract_sqrt_argument_view(ctx, e).is_some() {
            return true;
        }
        if extract_unary_log_argument_view(ctx, e).is_some() {
            return true;
        }
        match ctx.get(e) {
            // x^(p/q) where q is even requires x >= 0
            Expr::Pow(_base, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    is_even_root_exponent(n)
                } else {
                    false
                }
            }
            _ => false,
        }
    };

    // Check if LHS is provably positive AND RHS has structure that needs it
    let lhs_positive = crate::helpers::prove_positive(ctx, lhs, vd);
    if matches!(
        lhs_positive,
        crate::domain::Proof::Proven | crate::domain::Proof::ProvenImplicit
    ) {
        // Only propagate if RHS has structural constraints that benefit from positivity
        // This prevents false requires like "2*x + 3 > 0" when RHS is just a number
        if has_positivity_structure(ctx, rhs) {
            let rhs_check = crate::helpers::prove_positive(ctx, rhs, vd);
            if rhs_check != crate::domain::Proof::Disproven && !is_abs(ctx, rhs) {
                add_positive_and_propagate(ctx, rhs, &mut derived, vd);
            }
        }
    }

    // Check if RHS is provably positive AND LHS has structure that needs it
    let rhs_positive = crate::helpers::prove_positive(ctx, rhs, vd);
    if matches!(
        rhs_positive,
        crate::domain::Proof::Proven | crate::domain::Proof::ProvenImplicit
    ) {
        // Only propagate if LHS has structural constraints that benefit from positivity
        // This prevents false requires like "2*x + 3 > 0" when LHS is just a polynomial
        if has_positivity_structure(ctx, lhs) {
            let lhs_check = crate::helpers::prove_positive(ctx, lhs, vd);
            if lhs_check != crate::domain::Proof::Disproven && !is_abs(ctx, lhs) {
                add_positive_and_propagate(ctx, lhs, &mut derived, vd);
            }
        }
    }

    derived
}

/// Add Positive(expr) and propagate through sqrt/ln/abs structure.
/// - Positive(sqrt(t)) → Positive(t)
/// - Positive(abs(t)) → NonZero(t) (since |t| > 0 ⟺ t ≠ 0)
///
/// V2.3: Now takes ValueDomain to filter out conditions that are Disproven
/// (e.g., 0*x > 0 is always false, should not become a "Requires")
fn add_positive_and_propagate(
    ctx: &Context,
    expr: ExprId,
    derived: &mut Vec<ImplicitCondition>,
    vd: ValueDomain,
) {
    // V2.3: Skip adding conditions that are provably false (Disproven)
    // Example: 0*x > 0 is Disproven, so don't add it as a "Requires"
    let positive_check = crate::helpers::prove_positive(ctx, expr, vd);
    if positive_check == crate::domain::Proof::Disproven {
        return; // Don't add impossible conditions
    }

    match ctx.get(expr) {
        // abs(t) > 0 ⟺ t ≠ 0 (since abs is always ≥ 0)
        // Don't add Positive(abs(t)) - it's redundant and confusing
        // Instead add NonZero(t) which is the actual constraint
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 =>
        {
            derived.push(ImplicitCondition::NonZero(args[0]));
        }
        // sqrt(t) > 0 implies t > 0
        Expr::Function(_, _) if extract_sqrt_argument_view(ctx, expr).is_some() => {
            let Some(arg) = extract_sqrt_argument_view(ctx, expr) else {
                return;
            };
            derived.push(ImplicitCondition::Positive(expr));
            derived.push(ImplicitCondition::Positive(arg));
        }
        // t^(1/2) > 0 implies t > 0
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                if is_even_root_exponent(n) {
                    derived.push(ImplicitCondition::Positive(expr));
                    derived.push(ImplicitCondition::Positive(*base));
                    return;
                }
            }
            // Non-even-root power: add as-is
            derived.push(ImplicitCondition::Positive(expr));
        }
        _ => {
            // Default: add the base condition
            derived.push(ImplicitCondition::Positive(expr));
        }
    }
}

/// Check if expression contains any variables.
/// Uses iterative traversal to prevent stack overflow on deep expressions.
pub(crate) fn contains_variable(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Variable(_) => return true,
            Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => stack.push(*inner),
            Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
        }
    }

    false
}

/// Check if an expression is always non-negative for real values.
/// Returns true for patterns like:
/// - x² (x^2 for any even exponent)
/// - |x| (absolute value)
/// - x⁴, x⁶, etc. (any even power)
pub(crate) fn is_always_nonnegative(ctx: &Context, expr: ExprId) -> bool {
    // Use depth-limited version with max 50 levels to prevent stack overflow
    is_always_nonnegative_depth(ctx, expr, 50)
}

/// Internal is_always_nonnegative with explicit depth limit.
fn is_always_nonnegative_depth(ctx: &Context, expr: ExprId, depth: usize) -> bool {
    // Depth guard: return false if we've recursed too deep (conservative)
    if depth == 0 {
        return false;
    }

    match ctx.get(expr) {
        // Numeric constants: check if ≥ 0
        Expr::Number(n) => *n >= BigRational::from_integer(0.into()),

        // x^n where n is an even positive integer
        Expr::Pow(_base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                // Even integer exponent means always non-negative for real base
                if n.is_integer() {
                    let numer = n.numer();
                    if numer.is_even() && *numer > num_bigint::BigInt::from(0) {
                        return true;
                    }
                }
            }
            // Fallback: check if base is non-negative and exponent is positive
            false
        }

        // |x| is always non-negative
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 =>
        {
            true
        }

        // sqrt(x) is non-negative by definition (for real)
        Expr::Function(_, _) if extract_sqrt_argument_view(ctx, expr).is_some() => true,

        // x * x where both sides are the same = x², always non-negative
        Expr::Mul(l, r) => {
            if *l == *r {
                return true; // x * x = x²
            }
            // Product of two non-negatives is non-negative
            is_always_nonnegative_depth(ctx, *l, depth - 1)
                && is_always_nonnegative_depth(ctx, *r, depth - 1)
        }

        // Sum of non-negatives is non-negative
        Expr::Add(l, r) => {
            is_always_nonnegative_depth(ctx, *l, depth - 1)
                && is_always_nonnegative_depth(ctx, *r, depth - 1)
        }

        _ => false,
    }
}

/// Iterative domain inference (replaces recursive version).
/// Uses explicit stack to prevent stack overflow on deep expressions.
fn infer_recursive(ctx: &Context, root: ExprId, domain: &mut ImplicitDomain) {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            // sqrt(t) → NonNegative(t)
            // BUT skip numeric literals - they're trivially provable
            Expr::Function(_, _) if extract_sqrt_argument_view(ctx, expr).is_some() => {
                let Some(arg) = extract_sqrt_argument_view(ctx, expr) else {
                    continue;
                };
                if !matches!(ctx.get(arg), Expr::Number(_)) {
                    domain.add_nonnegative(arg);
                }
                stack.push(arg);
            }

            // ln(t) or log(t) → Positive(t)
            Expr::Function(_, _) if extract_unary_log_argument_view(ctx, expr).is_some() => {
                let Some(arg) = extract_unary_log_argument_view(ctx, expr) else {
                    continue;
                };
                domain.add_positive(arg);
                stack.push(arg);
            }

            // t^(1/2) or t^(p/q) where q is even → NonNegative(t)
            Expr::Pow(base, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if is_even_root_exponent(n) && !matches!(ctx.get(*base), Expr::Number(_)) {
                        domain.add_nonnegative(*base);
                    }
                }
                stack.push(*base);
                stack.push(*exp);
            }

            // Div(_, t) → NonZero(t)
            Expr::Div(num, den) => {
                domain.add_nonzero(*den);
                stack.push(*num);
                stack.push(*den);
            }

            // Process children
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

            // Leaf nodes: nothing to infer
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}

            Expr::Matrix { data, .. } => {
                stack.extend(data.iter().copied());
            }
            Expr::Hold(inner) => {
                stack.push(*inner);
            }
        }
    }
}

/// Check if an exponent represents an even root (e.g., 1/2, 1/4, 3/4).
pub(crate) fn is_even_root_exponent(n: &BigRational) -> bool {
    let denom = n.denom();
    // Check if denominator is even (2, 4, 6, ...)
    denom.is_even()
}
