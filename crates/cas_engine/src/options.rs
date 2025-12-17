//! Evaluation options and assumptions for the CAS engine.
//!
//! This module provides configuration for how expressions are evaluated,
//! particularly around branch assumptions for inverse functions.

/// Branch mode controls how inverse∘function compositions are simplified.
///
/// - `Strict` (default): Mathematically safe, never assumes domain restrictions
/// - `PrincipalBranch`: Educational mode, assumes principal domain for inverse trig
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

/// Assumptions for expression evaluation.
///
/// These are stored in `SessionState` and control simplification behavior.
#[derive(Clone, Debug, Default)]
pub struct Assumptions {
    /// How to handle inverse function compositions
    pub branch_mode: BranchMode,
    // Future: variable domain assumptions, positivity, etc.
}

impl Assumptions {
    /// Create assumptions with strict (safe) mode
    pub fn strict() -> Self {
        Self {
            branch_mode: BranchMode::Strict,
        }
    }

    /// Create assumptions with principal branch mode
    pub fn principal_branch() -> Self {
        Self {
            branch_mode: BranchMode::PrincipalBranch,
        }
    }
}
