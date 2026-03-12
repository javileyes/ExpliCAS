use super::*;

impl Repl {
    pub fn new() -> Self {
        let config = CasConfig::load();
        Self {
            core: cas_session::repl_api::build_repl_core_with_config(&config),
            verbosity: Verbosity::Normal,
            config,
        }
    }

    /// Build the REPL prompt with mode indicators.
    /// Only shows indicators for non-default modes to keep prompt clean.
    pub(crate) fn build_prompt(&self) -> String {
        cas_solver::session_api::lifecycle::build_repl_prompt(&self.core)
    }

    /// Converts function-style commands to command-style
    /// Examples:
    ///   simplify(...) -> simplify x^2 + 1
    ///   solve(...) -> solve x + 2 = 5, x
    pub(crate) fn preprocess_function_syntax(&self, line: &str) -> String {
        cas_solver::session_api::repl::preprocess_repl_function_syntax(line)
    }
}
