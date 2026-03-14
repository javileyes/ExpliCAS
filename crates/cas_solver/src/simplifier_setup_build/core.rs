use crate::rules::exponents::{EvaluatePowerRule, NegativeExponentNormalizationRule};
use crate::rules::grouping::CollectRule;
use crate::rules::logarithms::{EvaluateLogRule, ExponentialLogRule, SplitLogExponentsRule};
use crate::rules::trigonometry::{
    AngleIdentityRule, DoubleAngleRule, EvaluateTrigRule, PythagoreanIdentityRule, TanToSinCosRule,
    TrigHalfAngleSquaresRule,
};
use cas_solver_core::simplifier_config::SimplifierRuleConfig;

pub(super) fn add_core_rules(simplifier: &mut crate::Simplifier, config: &SimplifierRuleConfig) {
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
    simplifier.add_rule(Box::new(TrigHalfAngleSquaresRule));
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
    simplifier.add_rule(Box::new(NegativeExponentNormalizationRule));
    // Kept duplicated intentionally to preserve current behavior.
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    if config.log_split_exponents {
        simplifier.add_rule(Box::new(SplitLogExponentsRule));
    }
}
