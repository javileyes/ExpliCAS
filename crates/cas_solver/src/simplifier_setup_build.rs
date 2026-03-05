use crate::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule, MulZeroRule};
use crate::rules::calculus::{DiffRule, IntegrateRule};
use crate::rules::exponents::{
    EvaluatePowerRule, IdentityPowerRule, PowerPowerRule, PowerProductRule, PowerQuotientRule,
    ProductPowerRule,
};
use crate::rules::grouping::CollectRule;
use crate::rules::logarithms::{EvaluateLogRule, ExponentialLogRule, SplitLogExponentsRule};
use crate::rules::number_theory::NumberTheoryRule;
use crate::rules::polynomial::{
    AnnihilationRule, BinomialExpansionRule, CombineLikeTermsRule, DistributeRule,
};
use crate::rules::trigonometry::{
    AngleConsistencyRule, AngleIdentityRule, DoubleAngleRule, EvaluateTrigRule,
    PythagoreanIdentityRule, TanToSinCosRule,
};

use crate::{Simplifier, SimplifierRuleConfig};

/// Build a configured simplifier with the rule portfolio expected by CLI workflows.
///
/// This centralizes rule wiring outside frontends, so REPL/FFI/web can share
/// a consistent initialization path.
pub fn build_simplifier_with_rule_config(config: SimplifierRuleConfig) -> Simplifier {
    let mut simplifier = Simplifier::with_default_rules();

    // Always enabled core rules.
    simplifier.add_rule(Box::new(crate::rules::functions::AbsSquaredRule));
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    if config.trig_angle_sum {
        simplifier.add_rule(Box::new(AngleIdentityRule));
    }
    simplifier.add_rule(Box::new(TanToSinCosRule));
    if config.trig_double_angle {
        simplifier.add_rule(Box::new(DoubleAngleRule));
    }
    if config.canonicalize_trig_square {
        simplifier.add_rule(Box::new(
            crate::rules::trigonometry::CanonicalizeTrigSquareRule,
        ));
    }
    simplifier.add_rule(Box::new(EvaluateLogRule));
    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(crate::rules::algebra::SimplifyFractionRule));
    simplifier.add_rule(Box::new(crate::rules::algebra::ExpandRule));
    simplifier.add_rule(Box::new(crate::rules::algebra::ConservativeExpandRule));
    simplifier.add_rule(Box::new(crate::rules::algebra::FactorRule));
    simplifier.add_rule(Box::new(CollectRule));
    // Kept duplicated intentionally to preserve current behavior.
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    if config.log_split_exponents {
        simplifier.add_rule(Box::new(SplitLogExponentsRule));
    }

    // Advanced algebra rules (critical for solver).
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

    simplifier
}
