use super::{DeriveTargetForm, DeriveTargetProfile};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeriveStrategy {
    Simplify,
    IntegratePrep,
    SolvePrep,
    LogExpand,
    LogContract,
    SimplifyThenLogContract,
    TrigExpand,
    TrigContract,
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
            Self::Simplify => "simplify",
            Self::IntegratePrep => "integrate prep",
            Self::SolvePrep => "solve prep",
            Self::LogExpand => "expand_log",
            Self::LogContract => "contract logs",
            Self::SimplifyThenLogContract => "simplify -> contract logs",
            Self::TrigExpand => "expand trig",
            Self::TrigContract => "contract trig",
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

const TRIG_CONTRACT_TARGET_ORDER: &[DeriveStrategy] =
    &[DeriveStrategy::TrigContract, DeriveStrategy::Simplify];

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
