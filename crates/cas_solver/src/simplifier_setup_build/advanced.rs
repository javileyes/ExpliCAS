use crate::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule, MulZeroRule};
use crate::rules::calculus::{DiffRule, IntegrateRule};
use crate::rules::exponents::{
    IdentityPowerRule, PowerPowerRule, PowerProductRule, PowerQuotientRule, ProductPowerRule,
};
use crate::rules::number_theory::NumberTheoryRule;
use crate::rules::polynomial::{
    AnnihilationRule, BinomialExpansionRule, CombineLikeTermsRule, DistributeRule,
};
use crate::rules::trigonometry::AngleConsistencyRule;

pub(super) fn add_advanced_rules(
    simplifier: &mut crate::Simplifier,
    config: &crate::SimplifierRuleConfig,
) {
    simplifier.add_rule(Box::new(crate::rules::algebra::NestedFractionRule));
    simplifier.add_rule(Box::new(crate::rules::algebra::AddFractionsRule));
    simplifier.add_rule(Box::new(crate::rules::algebra::SimplifyMulDivRule));
    if config.rationalize_denominator {
        simplifier.add_rule(Box::new(crate::rules::algebra::RationalizeDenominatorRule));
    }
    simplifier.add_rule(Box::new(crate::rules::algebra::CancelCommonFactorsRule));

    if config.distribute {
        simplifier.add_rule(Box::new(DistributeRule));
    }
    if config.expand_binomials {
        simplifier.add_rule(Box::new(BinomialExpansionRule));
    }
    if config.factor_difference_squares {
        simplifier.add_rule(Box::new(crate::rules::algebra::FactorDifferenceSquaresRule));
    }
    if config.root_denesting {
        simplifier.add_rule(Box::new(crate::rules::algebra::RootDenestingRule));
    }
    if config.auto_factor {
        simplifier.add_rule(Box::new(crate::rules::algebra::AutomaticFactorRule));
    }

    simplifier.add_rule(Box::new(AngleConsistencyRule));
    // Kept duplicated intentionally to preserve current behavior.
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(PowerProductRule));
    simplifier.add_rule(Box::new(PowerQuotientRule));
    simplifier.add_rule(Box::new(IdentityPowerRule));
    simplifier.add_rule(Box::new(crate::rules::exponents::NegativeBasePowerRule));
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(MulZeroRule));
    simplifier.add_rule(Box::new(crate::rules::arithmetic::DivZeroRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(NumberTheoryRule));
}
