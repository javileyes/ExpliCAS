impl Default for super::SolverOptions {
    fn default() -> Self {
        Self {
            value_domain: crate::ValueDomain::RealOnly,
            domain_mode: crate::DomainMode::Generic,
            assume_scope: crate::AssumeScope::Real,
            budget: cas_solver_core::solve_budget::SolveBudget::default(),
            detailed_steps: true,
        }
    }
}
