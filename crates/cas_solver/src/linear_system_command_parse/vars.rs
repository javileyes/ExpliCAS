use crate::linear_system_command_types::LinearSystemSpecError;

pub(super) fn parse_linear_system_vars(
    var_parts: &[&str],
) -> Result<Vec<String>, LinearSystemSpecError> {
    let mut vars = Vec::with_capacity(var_parts.len());
    for var in var_parts {
        if !is_valid_linear_system_var(var) {
            return Err(LinearSystemSpecError::InvalidVariableName {
                name: (*var).to_string(),
            });
        }
        vars.push((*var).to_string());
    }
    Ok(vars)
}

fn is_valid_linear_system_var(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphabetic() || c == '_')
}
