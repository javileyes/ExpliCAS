mod power_rules;
mod rationalization;
mod simplification;

pub use power_rules::{
    EvaluatePowerRule, NegativeExponentNormalizationRule, PowerPowerRule, ProductPowerRule,
    ProductSameExponentRule, QuotientSameExponentRule, RootPowCancelRule,
};
pub use rationalization::{
    CubeRootDenRationalizeRule, PowPowCancelReciprocalRule, RationalizeLinearSqrtDenRule,
    RationalizeSumOfSqrtsDenRule, ReciprocalSqrtCanonRule, RootMergeDivRule, RootMergeMulRule,
};
pub use simplification::{
    EvenPowSubSwapRule, ExpQuotientRule, IdentityPowerRule, MulNaryCombinePowersRule,
    NegativeBasePowerRule, PowerProductRule, PowerQuotientRule,
};

pub fn register(simplifier: &mut crate::Simplifier) {
    // N-ary mul combine rule: handles (a*b)*a^2 → a^3*b
    simplifier.add_rule(Box::new(MulNaryCombinePowersRule));
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(ProductSameExponentRule));
    simplifier.add_rule(Box::new(QuotientSameExponentRule)); // a^n / b^n = (a/b)^n
                                                             // V2.14.45: RootPowCancelRule BEFORE PowerPowerRule for (x^n)^(1/n) with parity
    simplifier.add_rule(Box::new(RootPowCancelRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(NegativeExponentNormalizationRule)); // x^(-n) → 1/x^n
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(ExpQuotientRule)); // V2.14.45: e^a/e^b → e^(a-b)

    simplifier.add_rule(Box::new(IdentityPowerRule));
    simplifier.add_rule(Box::new(PowerProductRule));
    simplifier.add_rule(Box::new(PowerQuotientRule));
    simplifier.add_rule(Box::new(NegativeBasePowerRule));
    simplifier.add_rule(Box::new(EvenPowSubSwapRule)); // (b-a)^even → (a-b)^even
                                                       // Rationalize sqrt denominators: 1/(sqrt(t)+c) → (sqrt(t)-c)/(t-c²)
    simplifier.add_rule(Box::new(RationalizeLinearSqrtDenRule));
    // Rationalize sum of sqrts: 1/(sqrt(p)+sqrt(q)) → (sqrt(p)-sqrt(q))/(p-q)
    simplifier.add_rule(Box::new(RationalizeSumOfSqrtsDenRule));
    // Rationalize cube root: 1/(1+u^(1/3)) → (1-u^(1/3)+u^(2/3))/(1+u)
    simplifier.add_rule(Box::new(CubeRootDenRationalizeRule));
    // Merge sqrt products: sqrt(a)*sqrt(b) → sqrt(a*b) (with requires a≥0, b≥0)
    simplifier.add_rule(Box::new(RootMergeMulRule));
    // Merge sqrt quotients: sqrt(a)/sqrt(b) → sqrt(a/b) (with requires a≥0, b>0)
    simplifier.add_rule(Box::new(RootMergeDivRule));
    // Cancel reciprocal exponents: (u^y)^(1/y) → u (with requires u>0, y≠0)
    simplifier.add_rule(Box::new(PowPowCancelReciprocalRule));
    // Canonicalize reciprocal sqrt: 1/√x → x^(-1/2), √x/x → x^(-1/2)
    simplifier.add_rule(Box::new(ReciprocalSqrtCanonRule));
}

#[cfg(test)]
mod tests;
