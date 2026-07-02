//! Substitute command entrypoints exposed for CLI/frontends.

pub use crate::substitute::SubstituteSimplifyEvalOutput;
pub use crate::substitute::{substitute_power_aware, SubstituteOptions, SubstituteStrategy};
pub use crate::substitute_subcommand_eval::evaluate_substitute_subcommand;
pub use cas_api_models::{
    SubstituteCommandMode, SubstituteParseError, SubstituteRenderMode, SubstituteSubcommandOutput,
};
