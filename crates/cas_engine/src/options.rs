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

/// Evaluation options for expression processing.
///
/// Stored in `SessionState` for persistence, but can be overridden per request.
#[derive(Clone, Debug, Default)]
pub struct EvalOptions {
    /// How to handle inverse function compositions
    pub branch_mode: BranchMode,
    /// Which context-specific rules to enable
    pub context_mode: ContextMode,
}

impl EvalOptions {
    /// Create options with strict branch mode and standard context
    pub fn strict() -> Self {
        Self {
            branch_mode: BranchMode::Strict,
            context_mode: ContextMode::Standard,
        }
    }

    /// Create options with principal branch mode
    pub fn principal_branch() -> Self {
        Self {
            branch_mode: BranchMode::PrincipalBranch,
            context_mode: ContextMode::Auto,
        }
    }

    /// Create options for integration preparation
    pub fn integrate_prep() -> Self {
        Self {
            branch_mode: BranchMode::Strict,
            context_mode: ContextMode::IntegratePrep,
        }
    }

    /// Create options for equation solving
    pub fn solve() -> Self {
        Self {
            branch_mode: BranchMode::Strict,
            context_mode: ContextMode::Solve,
        }
    }
}

// Backwards compatibility alias
pub type Assumptions = EvalOptions;
