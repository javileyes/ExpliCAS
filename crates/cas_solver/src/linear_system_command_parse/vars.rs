use crate::linear_system_command_parse::LinearSystemSpecError;

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
    // Alphanumeric after an alphabetic head (D16, Fase 4 O3): constant names
    // like `C1`/`c2` are legitimate solve_system unknowns (the coupled
    // second-order IVP constants of O4 need them). A leading digit or any
    // operator character stays invalid.
    !s.is_empty()
        && s.chars()
            .next()
            .is_some_and(|c| c.is_alphabetic() || c == '_')
        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::is_valid_linear_system_var;

    #[test]
    fn constant_names_with_digits_are_valid_after_d16() {
        for name in ["c1", "C1", "C2", "k12", "x_1"] {
            assert!(is_valid_linear_system_var(name), "{name}");
        }
    }

    #[test]
    fn invalid_names_stay_invalid() {
        for name in ["1a", "2", "x-y", "x+y", "", "x y", "a.b"] {
            assert!(!is_valid_linear_system_var(name), "{name}");
        }
    }
}
