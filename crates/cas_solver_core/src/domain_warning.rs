/// A domain-assumption warning with source rule metadata.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DomainWarning {
    pub message: String,
    pub rule_name: String,
}

#[cfg(test)]
mod tests {
    use super::DomainWarning;
    use std::collections::HashSet;

    #[test]
    fn warning_is_hashable_for_dedup() {
        let a = DomainWarning {
            message: "m".to_string(),
            rule_name: "r".to_string(),
        };
        let b = DomainWarning {
            message: "m".to_string(),
            rule_name: "r".to_string(),
        };
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 1);
    }
}
