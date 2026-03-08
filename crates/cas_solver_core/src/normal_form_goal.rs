/// Goal for the current transformation, used to gate inverse rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum NormalFormGoal {
    /// Default: apply all simplification rules.
    #[default]
    Simplify,

    /// `expand()`: distribute products, don't collect terms.
    Expanded,

    /// `collect()`: group terms by variable, don't distribute.
    Collected,

    /// `factor()`: find common factors, don't expand.
    Factored,

    /// `expand_log()`: expand logarithms, don't contract.
    ExpandedLog,
}

#[cfg(test)]
mod tests {
    use super::NormalFormGoal;

    #[test]
    fn default_is_simplify() {
        assert_eq!(NormalFormGoal::default(), NormalFormGoal::Simplify);
    }
}
