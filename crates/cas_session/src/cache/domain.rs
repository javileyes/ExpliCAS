use cas_solver_core::domain_mode::DomainMode;

/// Session-local domain axis used by cache keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheDomainMode {
    Strict,
    Assume,
    Generic,
}

impl From<DomainMode> for CacheDomainMode {
    fn from(mode: DomainMode) -> Self {
        match mode {
            DomainMode::Strict => Self::Strict,
            DomainMode::Assume => Self::Assume,
            DomainMode::Generic => Self::Generic,
        }
    }
}
