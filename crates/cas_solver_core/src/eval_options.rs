//! Evaluation options for expression processing.

pub use crate::eval_option_axes::{
    AutoExpandBinomials, BranchMode, ComplexMode, ContextMode, HeuristicPoly, StepsMode,
};

/// Evaluation options for expression processing.
#[derive(Clone, Debug)]
pub struct EvalOptions {
    /// How to handle inverse function compositions.
    pub branch_mode: BranchMode,
    /// Whether to apply complex number rules.
    pub complex_mode: ComplexMode,
    /// Whether to collect simplification steps (runtime, not cached).
    pub steps_mode: StepsMode,
    /// Constant folding mode (Off or Safe).
    pub const_fold: crate::const_fold_types::ConstFoldMode,
    /// Whether to display blocked hints in REPL.
    pub hints_enabled: bool,
    /// Explain mode: show assumption summary after each evaluation.
    pub explain_mode: bool,
    /// Budget for Conditional branching in solve.
    pub budget: crate::solve_budget::SolveBudget,
    /// Verify solutions by substitution.
    pub check_solutions: bool,
    /// Display level for required conditions.
    pub requires_display: crate::domain_condition::RequiresDisplayLevel,
    /// Shared configuration (expand policy, semantics, context, etc.).
    pub shared: crate::simplify_options::SharedSemanticConfig,
}

impl EvalOptions {
    /// Create options with strict branch mode and standard context.
    pub fn strict() -> Self {
        Self {
            branch_mode: BranchMode::Strict,
            shared: crate::simplify_options::SharedSemanticConfig {
                context_mode: ContextMode::Standard,
                ..Default::default()
            },
            complex_mode: ComplexMode::Auto,
            steps_mode: StepsMode::On,
            ..Default::default()
        }
    }

    /// Create options with principal branch mode.
    pub fn principal_branch() -> Self {
        Self {
            branch_mode: BranchMode::PrincipalBranch,
            complex_mode: ComplexMode::Auto,
            steps_mode: StepsMode::On,
            ..Default::default()
        }
    }

    /// Create options for integration preparation.
    pub fn integrate_prep() -> Self {
        Self {
            branch_mode: BranchMode::Strict,
            shared: crate::simplify_options::SharedSemanticConfig {
                context_mode: ContextMode::IntegratePrep,
                ..Default::default()
            },
            complex_mode: ComplexMode::Auto,
            steps_mode: StepsMode::On,
            ..Default::default()
        }
    }

    /// Create options for equation solving.
    pub fn solve() -> Self {
        Self {
            branch_mode: BranchMode::Strict,
            shared: crate::simplify_options::SharedSemanticConfig {
                context_mode: ContextMode::Solve,
                ..Default::default()
            },
            complex_mode: ComplexMode::Auto,
            steps_mode: StepsMode::On,
            ..Default::default()
        }
    }

    /// Convert to SimplifyOptions for the simplification pipeline.
    pub fn to_simplify_options(&self) -> crate::simplify_options::SimplifyOptions {
        crate::simplify_options::SimplifyOptions {
            shared: self.shared.clone(),
            collect_steps: !matches!(self.steps_mode, StepsMode::Off),
            ..Default::default()
        }
    }
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            branch_mode: BranchMode::default(),
            complex_mode: ComplexMode::default(),
            steps_mode: StepsMode::default(),
            const_fold: crate::const_fold_types::ConstFoldMode::default(),
            hints_enabled: true,
            explain_mode: false,
            budget: crate::solve_budget::SolveBudget::default(),
            check_solutions: false,
            requires_display: crate::domain_condition::RequiresDisplayLevel::Essential,
            shared: crate::simplify_options::SharedSemanticConfig::default(),
        }
    }
}
