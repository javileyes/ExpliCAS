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
#[derive(Clone, Debug, Default)]
pub struct EvalOptions {
    /// How to handle inverse function compositions
    pub branch_mode: BranchMode,
    /// Which context-specific rules to enable
    pub context_mode: ContextMode,
    /// Whether to apply complex number rules
    pub complex_mode: ComplexMode,
    /// Whether to collect simplification steps (runtime, not cached)
    pub steps_mode: StepsMode,
}

impl EvalOptions {
    /// Create options with strict branch mode and standard context
    pub fn strict() -> Self {
        Self {
            branch_mode: BranchMode::Strict,
            context_mode: ContextMode::Standard,
            complex_mode: ComplexMode::Auto,
            steps_mode: StepsMode::On,
        }
    }

    /// Create options with principal branch mode
    pub fn principal_branch() -> Self {
        Self {
            branch_mode: BranchMode::PrincipalBranch,
            context_mode: ContextMode::Auto,
            complex_mode: ComplexMode::Auto,
            steps_mode: StepsMode::On,
        }
    }

    /// Create options for integration preparation
    pub fn integrate_prep() -> Self {
        Self {
            branch_mode: BranchMode::Strict,
            context_mode: ContextMode::IntegratePrep,
            complex_mode: ComplexMode::Auto,
            steps_mode: StepsMode::On,
        }
    }

    /// Create options for equation solving
    pub fn solve() -> Self {
        Self {
            branch_mode: BranchMode::Strict,
            context_mode: ContextMode::Solve,
            complex_mode: ComplexMode::Auto,
            steps_mode: StepsMode::On,
        }
    }
}

// Backwards compatibility alias
pub type Assumptions = EvalOptions;
