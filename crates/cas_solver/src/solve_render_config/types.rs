#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveCommandRenderConfig {
    pub show_steps: bool,
    pub show_verbose_substeps: bool,
    pub requires_display: crate::RequiresDisplayLevel,
    pub debug_mode: bool,
    pub hints_enabled: bool,
    pub domain_mode: crate::DomainMode,
    pub check_solutions: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveDisplayMode {
    None,
    Succinct,
    Normal,
    Verbose,
}
