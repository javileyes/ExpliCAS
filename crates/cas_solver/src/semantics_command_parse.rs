mod axis;
mod route;

use cas_api_models::SemanticsCommandInput;

/// Parse a full `semantics ...` command line.
pub fn parse_semantics_command_input(line: &str) -> SemanticsCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    route::parse_semantics_command_args(&args)
}
