mod equations;
mod invocation;
mod split;
mod vars;

use cas_ast::Context;

use crate::linear_system_command_types::{LinearSystemSpec, LinearSystemSpecError};

pub(crate) fn parse_linear_system_spec(
    ctx: &mut Context,
    input: &str,
) -> Result<LinearSystemSpec, LinearSystemSpecError> {
    let parts = split::split_linear_system_parts(input);

    if parts.len() < 4 || !parts.len().is_multiple_of(2) {
        return Err(LinearSystemSpecError::InvalidPartCount);
    }

    let n = parts.len() / 2;
    let eq_parts = &parts[0..n];
    let var_parts = &parts[n..2 * n];
    let vars = vars::parse_linear_system_vars(var_parts)?;
    let exprs = equations::parse_linear_system_exprs(ctx, eq_parts)?;

    Ok(LinearSystemSpec { exprs, vars })
}

pub fn parse_linear_system_invocation_input(line: &str) -> String {
    invocation::parse_linear_system_invocation_input(line)
}
