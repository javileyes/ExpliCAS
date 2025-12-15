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
    // Compact rationalization rules (Level 0, 1) - should apply first
    simplifier.add_rule(Box::new(RationalizeSingleSurdRule));
    simplifier.add_rule(Box::new(RationalizeBinomialSurdRule));
    // General rationalization rules (Level 2) - fallback for complex cases
    simplifier.add_rule(Box::new(RationalizeDenominatorRule));
    simplifier.add_rule(Box::new(GeneralizedRationalizationRule));
    simplifier.add_rule(Box::new(RationalizeProductDenominatorRule));
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(CancelCommonFactorsRule));
    simplifier.add_rule(Box::new(QuotientOfPowersRule));
    simplifier.add_rule(Box::new(RootDenestingRule));
    simplifier.add_rule(Box::new(SimplifySquareRootRule));
    simplifier.add_rule(Box::new(PullConstantFromFractionRule));
    simplifier.add_rule(Box::new(ExpandRule));
    simplifier.add_rule(Box::new(FactorBasedLCDRule));
    // P2: DifferenceOfSquaresRule for (a-b)(a+b) → a² - b²
    simplifier.add_rule(Box::new(DifferenceOfSquaresRule));
    // R1, R2: Fraction difference canonicalization for cyclic sums
    simplifier.add_rule(Box::new(AbsorbNegationIntoDifferenceRule));
    simplifier.add_rule(Box::new(CanonicalDifferenceProductRule));
    // Factor common integer from sums (POST phase): 2*√2 - 2 → 2*(√2 - 1)
    // Safe because DistributeRule now has PhaseMask excluding POST
    simplifier.add_rule(Box::new(FactorCommonIntegerFromAdd));
    // simplifier.add_rule(Box::new(FactorDifferenceSquaresRule)); // Too aggressive for default, causes loops with DistributeRule
}
