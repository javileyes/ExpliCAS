use num_rational::BigRational;

/// Information about a fraction sum that was computed.
#[derive(Debug)]
pub(crate) struct FractionSumInfo {
    /// The fractions that were summed.
    pub fractions: Vec<BigRational>,
    /// The result of the sum.
    pub result: BigRational,
}
