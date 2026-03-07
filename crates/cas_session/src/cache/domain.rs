/// Session-local domain axis used by cache keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheDomainMode {
    Strict,
    Assume,
    Generic,
}

impl From<cas_solver::DomainMode> for CacheDomainMode {
    fn from(mode: cas_solver::DomainMode) -> Self {
        match mode {
            cas_solver::DomainMode::Strict => Self::Strict,
            cas_solver::DomainMode::Assume => Self::Assume,
            cas_solver::DomainMode::Generic => Self::Generic,
        }
    }
}
