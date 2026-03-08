//! Shared option axes used by evaluation and simplification runtimes.

/// Branch mode controls how inverse-function compositions are simplified.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BranchMode {
    /// Safe mode: never assumes restricted principal domains.
    #[default]
    Strict,
    /// Educational mode: assumes principal domain when applying inverses.
    PrincipalBranch,
}

/// Context mode controls intent-specific simplification behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ContextMode {
    /// Auto-detect from expression.
    #[default]
    Auto,
    /// Default safe simplification.
    Standard,
    /// Preserve forms useful for solving.
    Solve,
    /// Enable transforms useful for integration preparation.
    IntegratePrep,
}

/// Complex mode controls whether complex-number rules are applied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ComplexMode {
    /// Enable complex rules if needed by expression shape.
    #[default]
    Auto,
    /// Never apply complex rules.
    Off,
    /// Always apply complex rules.
    On,
}

/// Controls automatic expansion of small standalone binomials.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AutoExpandBinomials {
    /// Never auto-expand standalone binomials.
    #[default]
    Off,
    /// Expand small binomials under budget limits.
    On,
}

/// Controls smart polynomial simplification in Add/Sub contexts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeuristicPoly {
    /// Only strict identity normalization.
    #[default]
    Off,
    /// Smart factor+normalize pipeline.
    On,
}

/// Controls whether simplification step traces are collected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StepsMode {
    /// Full steps with before/after expressions.
    #[default]
    On,
    /// No steps collected.
    Off,
    /// Compact steps (no before/after payload).
    Compact,
}

#[cfg(test)]
mod tests {
    use super::{
        AutoExpandBinomials, BranchMode, ComplexMode, ContextMode, HeuristicPoly, StepsMode,
    };

    #[test]
    fn defaults_are_stable() {
        assert_eq!(BranchMode::default(), BranchMode::Strict);
        assert_eq!(ContextMode::default(), ContextMode::Auto);
        assert_eq!(ComplexMode::default(), ComplexMode::Auto);
        assert_eq!(AutoExpandBinomials::default(), AutoExpandBinomials::Off);
        assert_eq!(HeuristicPoly::default(), HeuristicPoly::Off);
        assert_eq!(StepsMode::default(), StepsMode::On);
    }
}
