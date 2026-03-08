use crate::domain_mode::DomainMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveAssumptionSectionConfig {
    pub debug_mode: bool,
    pub hints_enabled: bool,
    pub domain_mode: DomainMode,
}
