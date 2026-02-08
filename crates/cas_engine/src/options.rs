//! Evaluation options for the CAS engine.
//!
//! This module provides configuration for how expressions are evaluated,
//! including branch assumptions and context-aware simplification modes.

/// Branch mode controls how inverse∘function compositions are simplified.
///
/// - `Strict` (default): Mathematically safe, never assumes domain restrictions
/// - `PrincipalBranch`: Educational mode, assumes principal domain for inverse trig
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BranchMode {
    /// Safe mode: inverse∘function compositions are not simplified
    /// because they are only valid on restricted domains.
    /// Example: `atan(tan(x))` remains as-is (correct for all x)
    #[default]
    Strict,

    /// Educational mode: assumes inputs are in principal domain.
    /// Simplifications like `atan(tan(u)) → u` are allowed,
    /// but emit domain warnings via `Rewrite::domain_assumption`.
    PrincipalBranch,
}

/// Context mode controls which set of rules are applied based on intent.
///
/// Different mathematical operations benefit from different transformations:
/// - Integration: product→sum, telescoping series
/// - Solving: preserve polynomial structure
/// - General simplification: conservative, universally valid
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ContextMode {
    /// Auto-detect from expression: if contains integral → IntegratePrep
    #[default]
    Auto,

    /// Default safe simplification loop (no context-specific transforms)
    Standard,

    /// Preserve forms useful for equation solving strategies
    Solve,

    /// Enable transformations useful for integration:
    /// - Product→sum for trigonometric products
    /// - Telescoping series expansion
    /// - Partial fractions preparation
    IntegratePrep,
}

/// Complex mode controls whether complex number rules are applied.
///
/// - `Auto`: Detect if expression contains `i`; if so, enable complex rules
/// - `Off`: Never apply complex rules (i remains literal)
/// - `On`: Always apply complex rules (i² = -1, gaussian arithmetic)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ComplexMode {
    /// Auto-detect: if expression contains `i`, enable complex rules
    #[default]
    Auto,

    /// Complex rules disabled: `i` is treated as a literal constant
    Off,

    /// Complex rules enabled: i² = -1, gaussian arithmetic
    On,
}

/// V2.15.8: Controls automatic binomial/polynomial expansion.
///
/// - `Off`: Never auto-expand standalone binomials (default)
/// - `On`: Always expand small binomials (subject to budget: n≤6, base≤3 terms)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AutoExpandBinomials {
    /// Never auto-expand standalone binomials
    #[default]
    Off,
    /// Always expand small binomials (subject to budget: n≤6, base≤3 terms)
    On,
}

/// V2.15.9: Controls smart polynomial simplification in Add/Sub contexts.
///
/// - `Off`: Only polynomial identity → 0 (default)
/// - `On`: Smart mode - extract common factor, then poly normalize
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeuristicPoly {
    /// Only PolynomialIdentityZeroRule active
    #[default]
    Off,
    /// Smart: HeuristicExtractCommonFactorAddRule + HeuristicPolyNormalizeAddRule
    On,
}

/// Steps mode controls whether simplification steps are collected.
///
/// - `On` (default): Full steps with before/after expressions
/// - `Off`: No steps (minimal allocations), but domain_warnings preserved
/// - `Compact`: Only rule_name, description, domain_assumption (no before/after)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StepsMode {
    /// Full steps with before/after local expressions
    #[default]
    On,

    /// No steps collected (fastest), but domain_warnings still computed
    Off,

    /// Compact steps: rule_name + description only, no before/after
    Compact,
}

/// Evaluation options for expression processing.
///
/// Stored in `SessionState` for persistence, but can be overridden per request.
#[derive(Clone, Debug)]
pub struct EvalOptions {
    /// How to handle inverse function compositions
    pub branch_mode: BranchMode,
    /// Which context-specific rules to enable
    pub context_mode: ContextMode,
    /// Whether to apply complex number rules
    pub complex_mode: ComplexMode,
    /// Whether to collect simplification steps (runtime, not cached)
    pub steps_mode: StepsMode,
    /// Auto-expand policy: Off (default) or Auto (expand cheap cases)
    pub expand_policy: crate::phase::ExpandPolicy,
    /// Budget for auto-expand (only used when expand_policy=Auto)
    pub expand_budget: crate::phase::ExpandBudget,
    /// Auto-expand policy for logarithms: Off (default) or Auto
    pub log_expand_policy: crate::phase::ExpandPolicy,
    /// Semantic configuration: domain_mode, value_domain, branch, inv_trig, assume_scope
    pub semantics: crate::semantics::EvalConfig,
    /// Constant folding mode (Off or Safe)
    pub const_fold: crate::const_fold::ConstFoldMode,
    /// Assumption reporting level (Off, Summary, Trace)
    pub assumption_reporting: crate::assumptions::AssumptionReporting,
    /// Whether to display blocked hints in REPL (default: true)
    pub hints_enabled: bool,
    /// Explain mode: show assumption summary after each evaluation (default: false)
    pub explain_mode: bool,
    /// V2.0: Budget for Conditional branching in solve
    pub budget: crate::solver::SolveBudget,
    /// Issue #5: Verify solutions by substitution (default: false)
    pub check_solutions: bool,
    /// Display level for required conditions (Essential hides witness-surviving requires)
    pub requires_display: crate::implicit_domain::RequiresDisplayLevel,
    /// V2.15.8: Auto-expand small binomial powers (education mode)
    pub autoexpand_binomials: AutoExpandBinomials,
    /// V2.15.9: Smart polynomial simplification in Add/Sub contexts
    pub heuristic_poly: HeuristicPoly,
}

impl EvalOptions {
    /// Create options with strict branch mode and standard context
    pub fn strict() -> Self {
        Self {
            branch_mode: BranchMode::Strict,
            context_mode: ContextMode::Standard,
            complex_mode: ComplexMode::Auto,
            steps_mode: StepsMode::On,
            ..Default::default()
        }
    }

    /// Create options with principal branch mode
    pub fn principal_branch() -> Self {
        Self {
            branch_mode: BranchMode::PrincipalBranch,
            context_mode: ContextMode::Auto,
            complex_mode: ComplexMode::Auto,
            steps_mode: StepsMode::On,
            ..Default::default()
        }
    }

    /// Create options for integration preparation
    pub fn integrate_prep() -> Self {
        Self {
            branch_mode: BranchMode::Strict,
            context_mode: ContextMode::IntegratePrep,
            complex_mode: ComplexMode::Auto,
            steps_mode: StepsMode::On,
            ..Default::default()
        }
    }

    /// Create options for equation solving
    pub fn solve() -> Self {
        Self {
            branch_mode: BranchMode::Strict,
            context_mode: ContextMode::Solve,
            complex_mode: ComplexMode::Auto,
            steps_mode: StepsMode::On,
            ..Default::default()
        }
    }

    /// Convert to SimplifyOptions for the pipeline.
    /// This bridges EvalOptions (REPL/Session level) to SimplifyOptions (pipeline level).
    pub fn to_simplify_options(&self) -> crate::phase::SimplifyOptions {
        crate::phase::SimplifyOptions {
            expand_policy: self.expand_policy,
            expand_budget: self.expand_budget,
            log_expand_policy: self.log_expand_policy,
            context_mode: self.context_mode,
            collect_steps: !matches!(self.steps_mode, StepsMode::Off),
            semantics: self.semantics,
            assumption_reporting: self.assumption_reporting,
            autoexpand_binomials: self.autoexpand_binomials, // V2.15.8
            heuristic_poly: self.heuristic_poly,             // V2.15.9
            ..Default::default()
        }
    }
}

// Backwards compatibility alias
pub type Assumptions = EvalOptions;

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            branch_mode: BranchMode::default(),
            context_mode: ContextMode::default(),
            complex_mode: ComplexMode::default(),
            steps_mode: StepsMode::default(),
            expand_policy: crate::phase::ExpandPolicy::default(),
            expand_budget: crate::phase::ExpandBudget::default(),
            log_expand_policy: crate::phase::ExpandPolicy::Off, // On by default to avoid surprises
            semantics: crate::semantics::EvalConfig::default(),
            const_fold: crate::const_fold::ConstFoldMode::default(),
            assumption_reporting: crate::assumptions::AssumptionReporting::default(),
            hints_enabled: true, // Pedagogical hints on by default
            explain_mode: false, // Explain mode off by default
            budget: crate::solver::SolveBudget::default(),
            check_solutions: false, // Solution verification off by default
            requires_display: crate::implicit_domain::RequiresDisplayLevel::Essential,
            autoexpand_binomials: AutoExpandBinomials::Off, // V2.15.8: On by default
            heuristic_poly: HeuristicPoly::On,              // V2.15.9: On by default
        }
    }
}
