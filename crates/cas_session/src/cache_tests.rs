#[cfg(test)]
mod tests {
    use crate::cache::{CacheDomainMode, SimplifyCacheKey};

    #[test]
    fn cache_key_from_domain_flag_maps_known_values() {
        assert_eq!(
            SimplifyCacheKey::from_domain_flag("strict").domain,
            CacheDomainMode::Strict
        );
        assert_eq!(
            SimplifyCacheKey::from_domain_flag("assume").domain,
            CacheDomainMode::Assume
        );
        assert_eq!(
            SimplifyCacheKey::from_domain_flag("generic").domain,
            CacheDomainMode::Generic
        );
    }

    #[test]
    fn cache_key_from_domain_flag_defaults_to_generic() {
        assert_eq!(
            SimplifyCacheKey::from_domain_flag("unknown").domain,
            CacheDomainMode::Generic
        );
    }
}
