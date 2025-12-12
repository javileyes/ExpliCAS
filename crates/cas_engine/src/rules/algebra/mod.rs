#[cfg(test)]
mod tests;

pub mod helpers;
pub use helpers::*;

pub mod fractions;
pub use fractions::*;

pub mod distribution;
pub use distribution::*;

pub mod factoring;
pub use factoring::*;

pub mod roots;
pub use roots::*;

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(NestedFractionRule));
    simplifier.add_rule(Box::new(SimplifyMulDivRule));
    simplifier.add_rule(Box::new(AddFractionsRule));
    simplifier.add_rule(Box::new(RationalizeDenominatorRule));
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(CancelCommonFactorsRule));
    simplifier.add_rule(Box::new(QuotientOfPowersRule));
    simplifier.add_rule(Box::new(RootDenestingRule));
    simplifier.add_rule(Box::new(SimplifySquareRootRule));
    simplifier.add_rule(Box::new(PullConstantFromFractionRule));
    simplifier.add_rule(Box::new(ExpandRule));
    simplifier.add_rule(Box::new(FactorBasedLCDRule));
    // simplifier.add_rule(Box::new(FactorDifferenceSquaresRule)); // Too aggressive for default, causes loops with DistributeRule
}
