/// Branch policy for multi-valued functions.
///
/// Currently only principal branch is supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum BranchPolicy {
    /// Use principal branch (for example, `log(-1) = i*pi`).
    #[default]
    Principal,
}

#[cfg(test)]
mod tests {
    use super::BranchPolicy;

    #[test]
    fn default_is_principal() {
        assert_eq!(BranchPolicy::default(), BranchPolicy::Principal);
    }
}
