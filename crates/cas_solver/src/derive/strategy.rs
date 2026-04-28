use super::{DeriveTargetForm, DeriveTargetProfile};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeriveStrategy {
    Planner,
    Simplify,
    IntegratePrep,
    SolvePrep,
    FiniteAggregate,
    NumberTheory,
    CombineLikeTerms,
    FactorialRewrite,
    InverseTrigRewrite,
    FractionCancel,
    NestedFraction,
    RadicalRewrite,
    ExponentialRewrite,
    LogInversePower,
    HyperbolicRewrite,
    TrigRewrite,
    LogExpand,
    LogContract,
    SimplifyThenLogContract,
    TrigExpand,
    TrigContract,
    SimplifyThenTrigContract,
    Rationalize,
    FractionExpand,
    FractionDecompose,
    FractionCombine,
    FactorWithDivision,
    SimplifyThenFactorWithDivision,
    PowerMerge,
    OddHalfPowerExpand,
    Expand,
    Collect,
    Factor,
    SimplifyThenLogExpand,
    SimplifyThenTrigExpand,
    SimplifyThenOddHalfPowerExpand,
    SimplifyThenExpand,
    SimplifyThenCollect,
    SimplifyThenFactor,
}

impl DeriveStrategy {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Planner => "planner",
            Self::Simplify => "simplify",
            Self::IntegratePrep => "integrate prep",
            Self::SolvePrep => "solve prep",
            Self::FiniteAggregate => "finite sums/products",
            Self::NumberTheory => "number theory",
            Self::CombineLikeTerms => "combine like terms",
            Self::FactorialRewrite => "rewrite factorials",
            Self::InverseTrigRewrite => "rewrite inverse trigs",
            Self::FractionCancel => "cancel fraction",
            Self::NestedFraction => "nested fraction",
            Self::RadicalRewrite => "rewrite radicals",
            Self::ExponentialRewrite => "rewrite exponentials",
            Self::LogInversePower => "log inverse power",
            Self::HyperbolicRewrite => "rewrite hyperbolics",
            Self::TrigRewrite => "rewrite trigs",
            Self::LogExpand => "expand_log",
            Self::LogContract => "contract logs",
            Self::SimplifyThenLogContract => "simplify -> contract logs",
            Self::TrigExpand => "expand trig",
            Self::TrigContract => "contract trig",
            Self::SimplifyThenTrigContract => "simplify -> contract trig",
            Self::Rationalize => "rationalize",
            Self::FractionExpand => "expand fraction",
            Self::FractionDecompose => "split fraction",
            Self::FractionCombine => "combine fraction",
            Self::FactorWithDivision => "factor out with division",
            Self::SimplifyThenFactorWithDivision => "simplify -> factor out with division",
            Self::PowerMerge => "combine powers",
            Self::OddHalfPowerExpand => "expand odd half power",
            Self::Expand => "expand",
            Self::Collect => "collect",
            Self::Factor => "factor",
            Self::SimplifyThenLogExpand => "simplify -> expand_log",
            Self::SimplifyThenTrigExpand => "simplify -> expand trig",
            Self::SimplifyThenOddHalfPowerExpand => "simplify -> expand odd half power",
            Self::SimplifyThenExpand => "simplify -> expand",
            Self::SimplifyThenCollect => "simplify -> collect",
            Self::SimplifyThenFactor => "simplify -> factor",
        }
    }
}

const DEFAULT_STRATEGY_ORDER: &[DeriveStrategy] = &[
    DeriveStrategy::Simplify,
    DeriveStrategy::LogExpand,
    DeriveStrategy::TrigExpand,
    DeriveStrategy::Rationalize,
    DeriveStrategy::FractionExpand,
    DeriveStrategy::Expand,
    DeriveStrategy::Collect,
    DeriveStrategy::FactorWithDivision,
    DeriveStrategy::SimplifyThenLogExpand,
    DeriveStrategy::SimplifyThenTrigExpand,
    DeriveStrategy::SimplifyThenExpand,
    DeriveStrategy::SimplifyThenCollect,
    DeriveStrategy::SimplifyThenFactorWithDivision,
];

const INTEGRATE_PREP_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::IntegratePrep, DeriveStrategy::Simplify];

const SOLVE_PREP_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::SolvePrep, DeriveStrategy::Simplify];

const FINITE_AGGREGATE_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::FiniteAggregate, DeriveStrategy::Simplify];

const COMBINE_LIKE_TERMS_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::CombineLikeTerms, DeriveStrategy::Simplify];

const FACTORIAL_REWRITE_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::FactorialRewrite, DeriveStrategy::Simplify];

const INVERSE_TRIG_REWRITE_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::InverseTrigRewrite, DeriveStrategy::Simplify];

const FRACTION_CANCEL_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::FractionCancel, DeriveStrategy::Simplify];

const NESTED_FRACTION_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::NestedFraction, DeriveStrategy::Simplify];

const RADICAL_REWRITE_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::RadicalRewrite, DeriveStrategy::Simplify];

const EXPONENTIAL_REWRITE_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::ExponentialRewrite, DeriveStrategy::Simplify];

const HYPERBOLIC_REWRITE_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::HyperbolicRewrite, DeriveStrategy::Simplify];

const TRIG_REWRITE_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::TrigRewrite, DeriveStrategy::Simplify];

const LOG_EXPAND_TARGET_ORDER: &[DeriveStrategy] = &[
    DeriveStrategy::LogExpand,
    DeriveStrategy::SimplifyThenLogExpand,
    DeriveStrategy::Simplify,
    DeriveStrategy::Expand,
    DeriveStrategy::SimplifyThenExpand,
];

const TRIG_EXPAND_TARGET_ORDER: &[DeriveStrategy] = &[
    DeriveStrategy::TrigExpand,
    DeriveStrategy::SimplifyThenTrigExpand,
    DeriveStrategy::Simplify,
    DeriveStrategy::Expand,
    DeriveStrategy::SimplifyThenExpand,
];

const TRIG_CONTRACT_TARGET_ORDER: &[DeriveStrategy] = &[
    DeriveStrategy::TrigContract,
    DeriveStrategy::SimplifyThenTrigContract,
    DeriveStrategy::Simplify,
];

const LOG_CONTRACT_TARGET_ORDER: &[DeriveStrategy] = &[
    DeriveStrategy::LogContract,
    DeriveStrategy::SimplifyThenLogContract,
    DeriveStrategy::Simplify,
];

const RATIONALIZED_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::Rationalize, DeriveStrategy::Simplify];

const FRACTION_EXPANDED_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::FractionExpand, DeriveStrategy::Simplify];

const FRACTION_DECOMPOSED_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::FractionDecompose, DeriveStrategy::Simplify];

const FRACTION_COMBINED_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::FractionCombine, DeriveStrategy::Simplify];

const FACTOR_WITH_DIVISION_TARGET_ORDER: &[DeriveStrategy] = &[
    DeriveStrategy::FactorWithDivision,
    DeriveStrategy::SimplifyThenFactorWithDivision,
    DeriveStrategy::Collect,
    DeriveStrategy::SimplifyThenCollect,
    DeriveStrategy::Simplify,
];

const POWER_MERGED_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::PowerMerge, DeriveStrategy::Simplify];

const ODD_HALF_POWER_TARGET_ORDER: &[DeriveStrategy] = &[
    DeriveStrategy::OddHalfPowerExpand,
    DeriveStrategy::SimplifyThenOddHalfPowerExpand,
    DeriveStrategy::Simplify,
];

const EXPAND_TARGET_ORDER: &[DeriveStrategy] = &[
    DeriveStrategy::Expand,
    DeriveStrategy::SimplifyThenExpand,
    DeriveStrategy::Simplify,
    DeriveStrategy::Collect,
    DeriveStrategy::SimplifyThenCollect,
];

const FACTOR_TARGET_ORDER: &[DeriveStrategy] = &[
    DeriveStrategy::Factor,
    DeriveStrategy::SimplifyThenFactor,
    DeriveStrategy::Simplify,
    DeriveStrategy::Collect,
    DeriveStrategy::SimplifyThenCollect,
];

const COLLECT_TARGET_ORDER: &[DeriveStrategy] = &[
    DeriveStrategy::Collect,
    DeriveStrategy::SimplifyThenCollect,
    DeriveStrategy::Simplify,
    DeriveStrategy::Factor,
    DeriveStrategy::SimplifyThenFactor,
];

pub(crate) fn ordered_strategies_for_target(
    profile: &DeriveTargetProfile,
) -> &'static [DeriveStrategy] {
    match &profile.form {
        DeriveTargetForm::IntegratePrepared => INTEGRATE_PREP_TARGET_ORDER,
        DeriveTargetForm::SolvePrepared => SOLVE_PREP_TARGET_ORDER,
        DeriveTargetForm::FiniteAggregateEvaluated => FINITE_AGGREGATE_TARGET_ORDER,
        DeriveTargetForm::LikeTermsCombined => COMBINE_LIKE_TERMS_TARGET_ORDER,
        DeriveTargetForm::FactorialRewritten => FACTORIAL_REWRITE_TARGET_ORDER,
        DeriveTargetForm::InverseTrigRewritten => INVERSE_TRIG_REWRITE_TARGET_ORDER,
        DeriveTargetForm::FractionCancelled => FRACTION_CANCEL_TARGET_ORDER,
        DeriveTargetForm::NestedFractionSimplified => NESTED_FRACTION_TARGET_ORDER,
        DeriveTargetForm::RadicalRewritten => RADICAL_REWRITE_TARGET_ORDER,
        DeriveTargetForm::ExponentialRewritten => EXPONENTIAL_REWRITE_TARGET_ORDER,
        DeriveTargetForm::HyperbolicRewritten => HYPERBOLIC_REWRITE_TARGET_ORDER,
        DeriveTargetForm::TrigRewritten => TRIG_REWRITE_TARGET_ORDER,
        DeriveTargetForm::LogExpanded => LOG_EXPAND_TARGET_ORDER,
        DeriveTargetForm::LogContracted => LOG_CONTRACT_TARGET_ORDER,
        DeriveTargetForm::TrigExpanded => TRIG_EXPAND_TARGET_ORDER,
        DeriveTargetForm::TrigContracted => TRIG_CONTRACT_TARGET_ORDER,
        DeriveTargetForm::Rationalized => RATIONALIZED_TARGET_ORDER,
        DeriveTargetForm::FractionExpanded => FRACTION_EXPANDED_TARGET_ORDER,
        DeriveTargetForm::FractionDecomposed => FRACTION_DECOMPOSED_TARGET_ORDER,
        DeriveTargetForm::FractionCombined => FRACTION_COMBINED_TARGET_ORDER,
        DeriveTargetForm::FactoredWithDivision { .. } => FACTOR_WITH_DIVISION_TARGET_ORDER,
        DeriveTargetForm::PowerMerged => POWER_MERGED_TARGET_ORDER,
        DeriveTargetForm::OddHalfPowerExpanded => ODD_HALF_POWER_TARGET_ORDER,
        DeriveTargetForm::Expanded => EXPAND_TARGET_ORDER,
        DeriveTargetForm::Collected { .. } => COLLECT_TARGET_ORDER,
        DeriveTargetForm::Factored => FACTOR_TARGET_ORDER,
        DeriveTargetForm::Simplified | DeriveTargetForm::Unknown => DEFAULT_STRATEGY_ORDER,
    }
}
