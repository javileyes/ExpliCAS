//! Shared simplification option models.

/// Shared configuration axes used by both EvalOptions and SimplifyOptions.
#[derive(Debug, Clone)]
pub struct SharedSemanticConfig {
    /// Auto-expand policy: Off (default) or Auto (expand cheap cases).
    pub expand_policy: crate::expand_policy::ExpandPolicy,
    /// Budget limits for auto-expansion (used when expand_policy=Auto).
    pub expand_budget: crate::expand_policy::ExpandBudget,
    /// Auto-expand policy for logarithms: Off (default) or Auto.
    pub log_expand_policy: crate::expand_policy::ExpandPolicy,
    /// Context mode (Standard, Solve, etc.) - Solve mode blocks auto-expand.
    pub context_mode: crate::eval_option_axes::ContextMode,
    /// Semantic configuration: domain_mode, value_domain, branch, inv_trig, assume_scope.
    pub semantics: crate::eval_config::EvalConfig,
    /// Assumption reporting level (Off, Summary, Trace).
    pub assumption_reporting: crate::assumption_reporting::AssumptionReporting,
    /// Auto-expand small binomial powers.
    pub autoexpand_binomials: crate::eval_option_axes::AutoExpandBinomials,
    /// Smart polynomial simplification in Add/Sub contexts.
    pub heuristic_poly: crate::eval_option_axes::HeuristicPoly,
}

impl Default for SharedSemanticConfig {
    fn default() -> Self {
        Self {
            expand_policy: crate::expand_policy::ExpandPolicy::default(),
            expand_budget: crate::expand_policy::ExpandBudget::default(),
            log_expand_policy: crate::expand_policy::ExpandPolicy::Off,
            context_mode: crate::eval_option_axes::ContextMode::default(),
            semantics: crate::eval_config::EvalConfig::default(),
            assumption_reporting: crate::assumption_reporting::AssumptionReporting::Off,
            autoexpand_binomials: crate::eval_option_axes::AutoExpandBinomials::Off,
            heuristic_poly: crate::eval_option_axes::HeuristicPoly::On,
        }
    }
}

/// Options controlling the simplification pipeline.
#[derive(Debug, Clone)]
pub struct SimplifyOptions {
    /// Whether to run the Transform phase (distribution, expansion).
    pub enable_transform: bool,
    /// Rationalization policy (includes auto_level and budgets).
    pub rationalize: cas_math::rationalize_policy::RationalizePolicy,
    /// Per-phase iteration budgets.
    pub budgets: crate::phase_budgets::PhaseBudgets,
    /// Whether to collect steps for timeline display.
    pub collect_steps: bool,
    /// Whether we're in "expand mode" (forces aggressive distribution).
    pub expand_mode: bool,
    /// Shared configuration (expand policy, semantics, context, etc.).
    pub shared: SharedSemanticConfig,
    /// Transformation goal: controls which inverse rules are gated out.
    pub goal: crate::normal_form_goal::NormalFormGoal,
    /// Purpose of simplification: Eval (default), SolvePrepass, or SolveTactic.
    pub simplify_purpose: crate::solve_safety_policy::SimplifyPurpose,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        Self {
            enable_transform: true,
            rationalize: cas_math::rationalize_policy::RationalizePolicy::default(),
            budgets: crate::phase_budgets::PhaseBudgets::default(),
            collect_steps: true,
            expand_mode: false,
            shared: SharedSemanticConfig::default(),
            goal: crate::normal_form_goal::NormalFormGoal::default(),
            simplify_purpose: crate::solve_safety_policy::SimplifyPurpose::default(),
        }
    }
}

impl SimplifyOptions {
    /// Options for `expand()` command: Core -> Transform -> PostCleanup (no Rationalize).
    pub fn for_expand() -> Self {
        let mut opt = Self::default();
        opt.rationalize.auto_level = cas_math::rationalize_policy::AutoRationalizeLevel::Off;
        opt.expand_mode = true;
        opt
    }

    /// Options for `simplify --no-transform`.
    pub fn no_transform() -> Self {
        Self {
            enable_transform: false,
            ..Default::default()
        }
    }

    /// Options for `simplify --no-rationalize`.
    pub fn no_rationalize() -> Self {
        let mut opt = Self::default();
        opt.rationalize.auto_level = cas_math::rationalize_policy::AutoRationalizeLevel::Off;
        opt
    }

    /// Options for solver pre-pass: only `SolveSafety::Always` rules.
    pub fn for_solve_prepass() -> Self {
        Self {
            simplify_purpose: crate::solve_safety_policy::SimplifyPurpose::SolvePrepass,
            collect_steps: false,
            ..Default::default()
        }
    }

    /// Options for solver tactic: conditional rules allowed based on DomainMode.
    pub fn for_solve_tactic(domain_mode: crate::domain_mode::DomainMode) -> Self {
        Self {
            simplify_purpose: crate::solve_safety_policy::SimplifyPurpose::SolveTactic,
            shared: SharedSemanticConfig {
                semantics: crate::eval_config::EvalConfig {
                    domain_mode,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }
}
