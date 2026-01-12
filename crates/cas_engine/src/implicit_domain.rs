//! Implicit Domain Inference.
//!
//! This module infers domain constraints that are implicitly required by
//! expression structure. For example, `sqrt(x)` in RealOnly mode implies `x ≥ 0`.
//!
//! # Key Concepts
//!
//! - **Implicit Domain**: Constraints derived from expression structure, not assumptions.
//! - **Witness**: The subexpression that enforces a constraint (e.g., `sqrt(x)` for `x ≥ 0`).
//! - **Witness Survival**: A constraint is only valid if its witness survives in the output.
//!
//! # Usage
//!
//! ```ignore
//! let implicit = infer_implicit_domain(ctx, root, ValueDomain::RealOnly);
//! // Later, when checking if x ≥ 0 is valid:
//! if implicit.contains_nonnegative(x) && witness_survives(ctx, x, output, WitnessKind::Sqrt) {
//!     // Can use ProvenImplicit
//! }
//! ```

use crate::semantics::ValueDomain;
use cas_ast::{Context, Expr, ExprId};
use num_integer::Integer;
use num_rational::BigRational;
use std::collections::HashSet;

// =============================================================================
// Implicit Condition Types
// =============================================================================

/// An implicit condition inferred from expression structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ImplicitCondition {
    /// x ≥ 0 (from sqrt(x) or x^(1/2))
    NonNegative(ExprId),
    /// x > 0 (from ln(x) or log(x))  
    Positive(ExprId),
    /// x ≠ 0 (from 1/x or Div(_, x))
    NonZero(ExprId),
}

impl ImplicitCondition {
    /// Human-readable display for REPL/UI.
    pub fn display(&self, ctx: &Context) -> String {
        use cas_ast::display::DisplayExpr;
        match self {
            ImplicitCondition::NonNegative(e) => {
                format!(
                    "{} ≥ 0",
                    DisplayExpr {
                        context: ctx,
                        id: *e
                    }
                )
            }
            ImplicitCondition::Positive(e) => {
                format!(
                    "{} > 0",
                    DisplayExpr {
                        context: ctx,
                        id: *e
                    }
                )
            }
            ImplicitCondition::NonZero(e) => {
                format!(
                    "{} ≠ 0",
                    DisplayExpr {
                        context: ctx,
                        id: *e
                    }
                )
            }
        }
    }

    /// Check if this condition is trivial (always true or on a constant expression).
    /// Trivial conditions like "2 > 0" or "x² ≥ 0" should be filtered from display.
    pub fn is_trivial(&self, ctx: &Context) -> bool {
        let expr = match self {
            ImplicitCondition::NonNegative(e) => *e,
            ImplicitCondition::Positive(e) => *e,
            ImplicitCondition::NonZero(e) => *e,
        };

        // Case 1: No variables = fully numeric constant (always trivial)
        if !contains_variable(ctx, expr) {
            return true;
        }

        // Case 2: For NonNegative, check if expression is always ≥ 0 (like x²)
        if let ImplicitCondition::NonNegative(e) = self {
            // Check for patterns that are always non-negative:
            // - x² (even power of variable)
            // - |x| (absolute value)
            // - x² + y² (sum of squares)
            if is_always_nonnegative(ctx, *e) {
                return true;
            }
        }

        false
    }

    /// Check if this condition's witness survives in the output expression.
    ///
    /// This is used for the "witness survival" display policy:
    /// - If `sqrt(x)` survives in output → `x ≥ 0` is implicitly shown (no need to display)
    /// - If `sqrt(x)` was consumed (e.g., `sqrt(x)^2 → x`) → `x ≥ 0` must be displayed
    pub fn witness_survives_in(&self, ctx: &Context, output: ExprId) -> bool {
        match self {
            ImplicitCondition::NonNegative(e) => {
                witness_survives(ctx, *e, output, WitnessKind::Sqrt)
            }
            ImplicitCondition::Positive(e) => witness_survives(ctx, *e, output, WitnessKind::Log),
            ImplicitCondition::NonZero(e) => {
                witness_survives(ctx, *e, output, WitnessKind::Division)
            }
        }
    }
}

// =============================================================================
// Condition Normalization (Canonical Form + Sign Normalization + Dedupe)
// =============================================================================

/// Normalize an expression for display in conditions.
///
/// This ensures that equivalent expressions like `1 + x` and `x + 1`
/// display consistently (as `x + 1` in polynomial order).
///
/// Strategy:
/// 1. Try polynomial conversion → get canonical polynomial form
/// 2. Normalize sign: if leading coefficient is negative, negate the whole expression
/// 3. Fallback: use DisplayExpr which already applies term ordering
///
/// The normalized ExprId is only used for DISPLAY - it does not affect the
/// underlying AST or comparisons.
pub fn normalize_condition_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    use crate::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};

    // Small budget - we're just normalizing for display, not doing heavy algebra
    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    // Try polynomial normalization
    if let Ok(poly) = multipoly_from_expr(ctx, expr, &budget) {
        // Check leading coefficient sign
        let needs_negation = if let Some((_coeff, _mono)) = poly.leading_term_lex() {
            // Leading coefficient is negative if coeff < 0
            _coeff < &num_rational::BigRational::from_integer(0.into())
        } else {
            false
        };

        // If leading coeff is negative, negate the polynomial
        let normalized_poly = if needs_negation { poly.neg() } else { poly };

        // Convert back to expression
        return multipoly_to_expr(&normalized_poly, ctx);
    }

    // Fallback: expression is not polynomial
    // Check if it's a Neg node and unwrap it
    if let Expr::Neg(inner) = ctx.get(expr) {
        // -E ≠ 0 is equivalent to E ≠ 0, prefer non-negated form
        return *inner;
    }

    // Otherwise return original (DisplayExpr will apply term ordering)
    expr
}

/// Normalize a condition for display (applies normalization to the inner expression).
pub fn normalize_condition(ctx: &mut Context, cond: &ImplicitCondition) -> ImplicitCondition {
    let normalized_expr = match cond {
        ImplicitCondition::NonNegative(e) => normalize_condition_expr(ctx, *e),
        ImplicitCondition::Positive(e) => normalize_condition_expr(ctx, *e),
        ImplicitCondition::NonZero(e) => normalize_condition_expr(ctx, *e),
    };

    match cond {
        ImplicitCondition::NonNegative(_) => ImplicitCondition::NonNegative(normalized_expr),
        ImplicitCondition::Positive(_) => ImplicitCondition::Positive(normalized_expr),
        ImplicitCondition::NonZero(_) => ImplicitCondition::NonZero(normalized_expr),
    }
}

/// Check if two conditions are equivalent (same type and polynomial-equivalent expressions).
fn conditions_equivalent(ctx: &Context, c1: &ImplicitCondition, c2: &ImplicitCondition) -> bool {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    // Must be same condition type
    let (e1, e2) = match (c1, c2) {
        (ImplicitCondition::NonNegative(a), ImplicitCondition::NonNegative(b)) => (*a, *b),
        (ImplicitCondition::Positive(a), ImplicitCondition::Positive(b)) => (*a, *b),
        (ImplicitCondition::NonZero(a), ImplicitCondition::NonZero(b)) => (*a, *b),
        _ => return false,
    };

    // Same ExprId = definitely equivalent
    if e1 == e2 {
        return true;
    }

    // Try polynomial comparison
    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    if let (Ok(p1), Ok(p2)) = (
        multipoly_from_expr(ctx, e1, &budget),
        multipoly_from_expr(ctx, e2, &budget),
    ) {
        // For NonZero, E ≠ 0 is equivalent to -E ≠ 0
        // So check if p1 == p2 OR p1 == -p2
        if p1 == p2 {
            return true;
        }

        // Check negation equivalence (E ≠ 0 ⟺ -E ≠ 0)
        // Also applies to Positive (E > 0 ⟺ -E < 0, but for reals we handle this)
        let p2_neg = p2.neg();
        if p1 == p2_neg {
            return true;
        }
    }

    false
}

/// Normalize and deduplicate a list of conditions for display.
///
/// This function:
/// 1. Normalizes each condition (canonical form + sign normalization)
/// 2. Deduplicates conditions by polynomial equivalence
/// 3. Preserves order (stable dedupe - keeps first occurrence)
///
/// Used when rendering "Requires:" in timeline and REPL.
pub fn normalize_and_dedupe_conditions(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
) -> Vec<ImplicitCondition> {
    let mut result: Vec<ImplicitCondition> = Vec::new();

    for cond in conditions {
        let normalized = normalize_condition(ctx, cond);

        // Check if we already have an equivalent condition
        let is_duplicate = result
            .iter()
            .any(|existing| conditions_equivalent(ctx, existing, &normalized));

        if !is_duplicate {
            result.push(normalized);
        }
    }

    result
}

/// Render conditions for display, applying normalization and deduplication.
///
/// This is the main entry point for rendering "Requires:" lists.
pub fn render_conditions_normalized(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
) -> Vec<String> {
    let normalized = normalize_and_dedupe_conditions(ctx, conditions);
    normalized.iter().map(|c| c.display(ctx)).collect()
}

// =============================================================================
// Display Level for Requires
// =============================================================================

/// Display level for required conditions.
///
/// Controls how many requires are shown to the user:
/// - `Essential`: Only show requires whose witness was consumed (pedagogically important)
/// - `All`: Show all requires including those with surviving witnesses
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequiresDisplayLevel {
    /// Show only essential requires (witness consumed or from equation derivation)
    #[default]
    Essential,
    /// Show all requires including implicit ones (witness survives)
    All,
}

/// Filter required conditions based on display level and witness survival.
///
/// In `Essential` mode, only shows requires whose witness was consumed:
/// - `sqrt(x) + 1` → x≥0 witness survives → HIDE
/// - `sqrt(x)^2 → x` → x≥0 witness consumed → SHOW
///
/// In `All` mode, shows everything.
///
/// # Arguments
/// * `requires` - All required conditions
/// * `ctx` - AST context  
/// * `result` - The result expression (to check witness survival)
/// * `level` - Display level (Essential or All)
///
/// # Returns
/// Filtered list of conditions to display
pub fn filter_requires_for_display<'a>(
    requires: &'a [ImplicitCondition],
    ctx: &Context,
    result: ExprId,
    level: RequiresDisplayLevel,
) -> Vec<&'a ImplicitCondition> {
    requires
        .iter()
        .filter(|cond| {
            // Always show if level is All
            if level == RequiresDisplayLevel::All {
                return true;
            }

            // Essential: show only if witness does NOT survive
            !cond.witness_survives_in(ctx, result)
        })
        .collect()
}

/// Set of implicit conditions inferred from an expression.
#[derive(Debug, Clone, Default)]
pub struct ImplicitDomain {
    conditions: HashSet<ImplicitCondition>,
}

impl ImplicitDomain {
    /// Create an empty implicit domain.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if domain is empty.
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Check if an expression has an implicit NonNegative constraint.
    pub fn contains_nonnegative(&self, expr: ExprId) -> bool {
        self.conditions
            .contains(&ImplicitCondition::NonNegative(expr))
    }

    /// Check if an expression has an implicit Positive constraint.
    pub fn contains_positive(&self, expr: ExprId) -> bool {
        self.conditions.contains(&ImplicitCondition::Positive(expr))
    }

    /// Check if an expression has an implicit NonZero constraint.
    pub fn contains_nonzero(&self, expr: ExprId) -> bool {
        self.conditions.contains(&ImplicitCondition::NonZero(expr))
    }

    /// Add a NonNegative condition.
    fn add_nonnegative(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::NonNegative(expr));
    }

    /// Add a Positive condition.
    fn add_positive(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::Positive(expr));
    }

    /// Add a NonZero condition.
    fn add_nonzero(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::NonZero(expr));
    }

    /// Get all conditions (for iteration/comparison)
    pub fn conditions(&self) -> &HashSet<ImplicitCondition> {
        &self.conditions
    }

    /// Get mutable access to conditions
    pub fn conditions_mut(&mut self) -> &mut HashSet<ImplicitCondition> {
        &mut self.conditions
    }

    /// Check if this domain is a superset of another (contains all its conditions)
    pub fn contains_all(&self, other: &ImplicitDomain) -> bool {
        other.conditions.is_subset(&self.conditions)
    }

    /// Get conditions that are in self but not in other (dropped conditions)
    pub fn dropped_from<'a>(&'a self, other: &'a ImplicitDomain) -> Vec<&'a ImplicitCondition> {
        self.conditions.difference(&other.conditions).collect()
    }

    /// Merge conditions from another domain into this one
    pub fn extend(&mut self, other: &ImplicitDomain) {
        for cond in &other.conditions {
            self.conditions.insert(cond.clone());
        }
    }

    /// Convert to a ConditionSet for solver use
    pub fn to_condition_set(&self) -> cas_ast::ConditionSet {
        let predicates: Vec<cas_ast::ConditionPredicate> =
            self.conditions.iter().map(|c| c.into()).collect();
        cas_ast::ConditionSet::from_predicates(predicates)
    }
}

// =============================================================================
// Adapters: ImplicitCondition ↔ ConditionPredicate
// =============================================================================

impl From<&ImplicitCondition> for cas_ast::ConditionPredicate {
    fn from(cond: &ImplicitCondition) -> Self {
        match cond {
            ImplicitCondition::NonNegative(e) => cas_ast::ConditionPredicate::NonNegative(*e),
            ImplicitCondition::Positive(e) => cas_ast::ConditionPredicate::Positive(*e),
            ImplicitCondition::NonZero(e) => cas_ast::ConditionPredicate::NonZero(*e),
        }
    }
}

impl From<ImplicitCondition> for cas_ast::ConditionPredicate {
    fn from(cond: ImplicitCondition) -> Self {
        (&cond).into()
    }
}

impl TryFrom<&cas_ast::ConditionPredicate> for ImplicitCondition {
    type Error = ();

    fn try_from(pred: &cas_ast::ConditionPredicate) -> Result<Self, Self::Error> {
        match pred {
            cas_ast::ConditionPredicate::NonNegative(e) => Ok(ImplicitCondition::NonNegative(*e)),
            cas_ast::ConditionPredicate::Positive(e) => Ok(ImplicitCondition::Positive(*e)),
            cas_ast::ConditionPredicate::NonZero(e) => Ok(ImplicitCondition::NonZero(*e)),
            _ => Err(()), // Defined, InvTrigPrincipalRange, EqZero, EqOne not mapped
        }
    }
}

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
fn format_expr_short(ctx: &Context, expr: ExprId) -> String {
    match ctx.get(expr) {
        Expr::Variable(name) => name.clone(),
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
        matches!(ctx.get(e), Expr::Function(name, args) if name == "abs" && args.len() == 1)
    };

    // Check if LHS is provably positive
    let lhs_positive = crate::helpers::prove_positive(ctx, lhs, vd);
    if matches!(
        lhs_positive,
        crate::domain::Proof::Proven | crate::domain::Proof::ProvenImplicit
    ) {
        // LHS > 0 proven, so RHS > 0 (for the equation to have solutions)
        // BUT: Only add this if RHS is not DISPROVEN (provably ≤ 0)
        // ALSO: Skip if RHS is abs(...) - solutions automatically satisfy abs(A) > 0
        let rhs_check = crate::helpers::prove_positive(ctx, rhs, vd);
        if rhs_check != crate::domain::Proof::Disproven && !is_abs(ctx, rhs) {
            add_positive_and_propagate(ctx, rhs, &mut derived, vd);
        }
    }

    // Check if RHS is provably positive
    let rhs_positive = crate::helpers::prove_positive(ctx, rhs, vd);
    if matches!(
        rhs_positive,
        crate::domain::Proof::Proven | crate::domain::Proof::ProvenImplicit
    ) {
        // RHS > 0 proven, so LHS > 0 (for the equation to have solutions)
        // Skip if LHS is abs(...) - solutions automatically satisfy abs(A) > 0
        let lhs_check = crate::helpers::prove_positive(ctx, lhs, vd);
        if lhs_check != crate::domain::Proof::Disproven && !is_abs(ctx, lhs) {
            add_positive_and_propagate(ctx, lhs, &mut derived, vd);
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
        Expr::Function(name, args) if name == "abs" && args.len() == 1 => {
            derived.push(ImplicitCondition::NonZero(args[0]));
        }
        // sqrt(t) > 0 implies t > 0
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
            derived.push(ImplicitCondition::Positive(expr));
            derived.push(ImplicitCondition::Positive(args[0]));
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
fn contains_variable(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Variable(_) => true,
        Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            contains_variable(ctx, *l) || contains_variable(ctx, *r)
        }
        Expr::Neg(inner) => contains_variable(ctx, *inner),
        Expr::Function(_, args) => args.iter().any(|a| contains_variable(ctx, *a)),
        Expr::Matrix { data, .. } => data.iter().any(|e| contains_variable(ctx, *e)),
    }
}

/// Check if an expression is always non-negative for real values.
/// Returns true for patterns like:
/// - x² (x^2 for any even exponent)
/// - |x| (absolute value)
/// - x⁴, x⁶, etc. (any even power)
fn is_always_nonnegative(ctx: &Context, expr: ExprId) -> bool {
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
        Expr::Function(name, args) if name == "abs" && args.len() == 1 => true,

        // sqrt(x) is non-negative by definition (for real)
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => true,

        // x * x where both sides are the same = x², always non-negative
        Expr::Mul(l, r) => {
            if *l == *r {
                return true; // x * x = x²
            }
            // Product of two non-negatives is non-negative
            is_always_nonnegative(ctx, *l) && is_always_nonnegative(ctx, *r)
        }

        // Sum of non-negatives is non-negative
        Expr::Add(l, r) => is_always_nonnegative(ctx, *l) && is_always_nonnegative(ctx, *r),

        _ => false,
    }
}

fn infer_recursive(ctx: &Context, expr: ExprId, domain: &mut ImplicitDomain) {
    match ctx.get(expr) {
        // sqrt(t) → NonNegative(t)
        // BUT skip numeric literals - they're trivially provable
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
            if !matches!(ctx.get(args[0]), Expr::Number(_)) {
                domain.add_nonnegative(args[0]);
            }
            infer_recursive(ctx, args[0], domain);
        }

        // ln(t) or log(t) → Positive(t)
        Expr::Function(name, args) if (name == "ln" || name == "log") && args.len() == 1 => {
            domain.add_positive(args[0]);
            infer_recursive(ctx, args[0], domain);
        }

        // t^(1/2) or t^(p/q) where q is even → NonNegative(t)
        // BUT skip numeric literals - they're trivially provable and don't need implicit domain protection
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                // Check if denominator is 2 (sqrt) or any even number (even root)
                if is_even_root_exponent(n) {
                    // Only add constraint for non-numeric bases (variables, expressions)
                    if !matches!(ctx.get(*base), Expr::Number(_)) {
                        domain.add_nonnegative(*base);
                    }
                }
            }
            infer_recursive(ctx, *base, domain);
            infer_recursive(ctx, *exp, domain);
        }

        // Div(_, t) → NonZero(t)
        Expr::Div(num, den) => {
            domain.add_nonzero(*den);
            infer_recursive(ctx, *num, domain);
            infer_recursive(ctx, *den, domain);
        }

        // Recursively process children
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            infer_recursive(ctx, *l, domain);
            infer_recursive(ctx, *r, domain);
        }
        Expr::Neg(inner) => {
            infer_recursive(ctx, *inner, domain);
        }
        Expr::Function(_, args) => {
            for arg in args {
                infer_recursive(ctx, *arg, domain);
            }
        }

        // Leaf nodes: nothing to infer
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}

        Expr::Matrix { data, .. } => {
            for elem in data {
                infer_recursive(ctx, *elem, domain);
            }
        }
    }
}

/// Check if an exponent represents an even root (e.g., 1/2, 1/4, 3/4).
fn is_even_root_exponent(n: &BigRational) -> bool {
    let denom = n.denom();
    // Check if denominator is even (2, 4, 6, ...)
    denom.is_even()
}

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

fn search_witness(ctx: &Context, target: ExprId, expr: ExprId, kind: WitnessKind) -> bool {
    match ctx.get(expr) {
        // Check if this node is a witness
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
            if kind == WitnessKind::Sqrt && exprs_equal(ctx, args[0], target) {
                return true;
            }
            search_witness(ctx, target, args[0], kind)
        }

        Expr::Function(name, args) if (name == "ln" || name == "log") && args.len() == 1 => {
            if kind == WitnessKind::Log && exprs_equal(ctx, args[0], target) {
                return true;
            }
            search_witness(ctx, target, args[0], kind)
        }

        Expr::Pow(base, exp) => {
            // Check for t^(1/2) form as witness for sqrt
            if kind == WitnessKind::Sqrt {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if is_even_root_exponent(n) && exprs_equal(ctx, *base, target) {
                        return true;
                    }
                }
            }
            search_witness(ctx, target, *base, kind) || search_witness(ctx, target, *exp, kind)
        }

        Expr::Div(num, den) => {
            if kind == WitnessKind::Division && exprs_equal(ctx, *den, target) {
                return true;
            }
            search_witness(ctx, target, *num, kind) || search_witness(ctx, target, *den, kind)
        }

        // Recursively search children
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            search_witness(ctx, target, *l, kind) || search_witness(ctx, target, *r, kind)
        }
        Expr::Neg(inner) => search_witness(ctx, target, *inner, kind),
        Expr::Function(_, args) => args.iter().any(|a| search_witness(ctx, target, *a, kind)),

        // Leaf nodes
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,

        Expr::Matrix { data, .. } => data.iter().any(|e| search_witness(ctx, target, *e, kind)),
    }
}

/// Check if two expressions are equal (by ExprId or structural comparison).
fn exprs_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
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

fn search_witness_in_context(
    ctx: &Context,
    target: ExprId,
    expr: ExprId,
    replaced_node: ExprId,
    replacement: Option<ExprId>,
    kind: WitnessKind,
) -> bool {
    // If we've reached the replaced node, search in replacement instead (if provided)
    if expr == replaced_node {
        if let Some(repl) = replacement {
            return search_witness(ctx, target, repl, kind);
        } else {
            // No replacement provided, skip this subtree
            return false;
        }
    }

    match ctx.get(expr) {
        // Check if this node is a witness
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
            if kind == WitnessKind::Sqrt && exprs_equal(ctx, args[0], target) {
                return true;
            }
            search_witness_in_context(ctx, target, args[0], replaced_node, replacement, kind)
        }

        Expr::Function(name, args) if (name == "ln" || name == "log") && args.len() == 1 => {
            if kind == WitnessKind::Log && exprs_equal(ctx, args[0], target) {
                return true;
            }
            search_witness_in_context(ctx, target, args[0], replaced_node, replacement, kind)
        }

        Expr::Pow(base, exp) => {
            // Check for t^(1/2) form as witness for sqrt
            if kind == WitnessKind::Sqrt {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if is_even_root_exponent(n) && exprs_equal(ctx, *base, target) {
                        return true;
                    }
                }
            }
            search_witness_in_context(ctx, target, *base, replaced_node, replacement, kind)
                || search_witness_in_context(ctx, target, *exp, replaced_node, replacement, kind)
        }

        Expr::Div(num, den) => {
            if kind == WitnessKind::Division && exprs_equal(ctx, *den, target) {
                return true;
            }
            search_witness_in_context(ctx, target, *num, replaced_node, replacement, kind)
                || search_witness_in_context(ctx, target, *den, replaced_node, replacement, kind)
        }

        // Recursively search children
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            search_witness_in_context(ctx, target, *l, replaced_node, replacement, kind)
                || search_witness_in_context(ctx, target, *r, replaced_node, replacement, kind)
        }
        Expr::Neg(inner) => {
            search_witness_in_context(ctx, target, *inner, replaced_node, replacement, kind)
        }
        Expr::Function(_, args) => args
            .iter()
            .any(|a| search_witness_in_context(ctx, target, *a, replaced_node, replacement, kind)),

        // Leaf nodes
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,

        Expr::Matrix { data, .. } => data
            .iter()
            .any(|e| search_witness_in_context(ctx, target, *e, replaced_node, replacement, kind)),
    }
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
    pub fn is_condition_implied(&self, ctx: &Context, cond: &ImplicitCondition) -> bool {
        // Check against global requires
        for known in &self.global_requires {
            if conditions_equivalent(ctx, cond, known) {
                return true;
            }
        }
        // Check against introduced requires
        for known in &self.introduced_requires {
            if conditions_equivalent(ctx, cond, known) {
                return true;
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

    // Rule 1: Branch/Heuristic/Domain never get reclassified
    match event.kind {
        AssumptionKind::BranchChoice
        | AssumptionKind::HeuristicAssumption
        | AssumptionKind::DomainExtension => {
            return (event.kind, None);
        }
        _ => {}
    }

    // Try to convert the event to an ImplicitCondition
    let implicit_cond = assumption_to_condition(event);

    match implicit_cond {
        Some(cond) => {
            // Rule 2: Check if implied
            if dc.is_condition_implied(ctx, &cond) {
                (AssumptionKind::DerivedFromRequires, None)
            } else {
                // Rule 3: Not implied → promote to RequiresIntroduced
                (AssumptionKind::RequiresIntroduced, Some(cond))
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
    use super::*;
    use cas_ast::Context;

    #[test]
    fn test_infer_sqrt_implies_nonnegative() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));

        let domain = infer_implicit_domain(&ctx, sqrt_x, ValueDomain::RealOnly);

        assert!(domain.contains_nonnegative(x));
        assert!(!domain.contains_positive(x));
    }

    #[test]
    fn test_infer_ln_implies_positive() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let ln_x = ctx.add(Expr::Function("ln".to_string(), vec![x]));

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
        let sqrt_x = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));
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
        let sqrt_x = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));

        let domain = infer_implicit_domain(&ctx, sqrt_x, ValueDomain::ComplexEnabled);

        assert!(domain.is_empty());
    }

    #[test]
    fn test_domain_delta_sqrt_square_to_x() {
        // sqrt(x)^2 -> x should be detected as ExpandsAnalytic
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));
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
        let sqrt_x = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));
        let y = ctx.var("y");
        let sqrt_y = ctx.add(Expr::Function("sqrt".to_string(), vec![y]));

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
}
