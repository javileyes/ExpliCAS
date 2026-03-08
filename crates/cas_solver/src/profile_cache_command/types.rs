pub use cas_solver_core::profile_cache_command_types::ProfileCacheCommandResult;

pub(super) enum ProfileCacheCommandInput {
    Status,
    Clear,
    Unknown(String),
}
