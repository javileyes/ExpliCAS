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

pub mod poly_gcd;
pub use poly_gcd::*;

pub mod gcd_exact;
pub use gcd_exact::*;

pub mod gcd_modp;
pub use gcd_modp::*;

pub mod poly_arith_modp;
pub use poly_arith_modp::*;

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(NestedFractionRule));
    simplifier.add_rule(Box::new(SimplifyMulDivRule));
    simplifier.add_rule(Box::new(AddFractionsRule));
    simplifier.add_rule(Box::new(CombineSameDenominatorFractionsRule));
    // Compact rationalization rules (Level 0, 1) - should apply first
    simplifier.add_rule(Box::new(RationalizeSingleSurdRule));
    simplifier.add_rule(Box::new(RationalizeBinomialSurdRule));
    // General rationalization rules (Level 2) - fallback for complex cases
    simplifier.add_rule(Box::new(RationalizeDenominatorRule)); // sqrt only (diff squares)
    simplifier.add_rule(Box::new(RationalizeNthRootBinomialRule)); // cube root and higher (geometric sum)
    simplifier.add_rule(Box::new(GeneralizedRationalizationRule));
    simplifier.add_rule(Box::new(RationalizeProductDenominatorRule));
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(CancelCommonFactorsRule));
    simplifier.add_rule(Box::new(QuotientOfPowersRule));
    simplifier.add_rule(Box::new(CancelNthRootBinomialFactorRule)); // (x+1)/(x^(1/3)+1) → x^(2/3)-x^(1/3)+1
    simplifier.add_rule(Box::new(SqrtConjugateCollapseRule)); // sqrt(A)*B → sqrt(B) when A*B=1
    simplifier.add_rule(Box::new(RootDenestingRule));
    simplifier.add_rule(Box::new(CubicConjugateTrapRule));
    simplifier.add_rule(Box::new(DenestSqrtAddSqrtRule));
    simplifier.add_rule(Box::new(DenestPerfectCubeInQuadraticFieldRule));
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
    // Polynomial GCD: poly_gcd(a*g, b*g) → g (structural)
    simplifier.add_rule(Box::new(PolyGcdRule));
    // Polynomial GCD exact: poly_gcd_exact(a, b) → algebraic GCD over ℚ
    simplifier.add_rule(Box::new(PolyGcdExactRule));
    // Polynomial GCD mod p: poly_gcd_modp(a, b) → fast Zippel GCD
    simplifier.add_rule(Box::new(PolyGcdModpRule));
    // Polynomial equality mod p: poly_eq_modp(a, b) → 1 or 0
    simplifier.add_rule(Box::new(PolyEqModpRule));
    // Polynomial arithmetic on __hold: __hold(P) - __hold(Q) = 0 if equal mod p
    simplifier.add_rule(Box::new(PolySubModpRule));
    // simplifier.add_rule(Box::new(FactorDifferenceSquaresRule)); // Too aggressive for default, causes loops with DistributeRule
}
