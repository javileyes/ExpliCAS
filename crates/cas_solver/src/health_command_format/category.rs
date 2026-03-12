use cas_solver_core::health_runtime::HealthStatusInput;

pub(super) fn resolve_health_category_filter<T, E, F>(
    status: &HealthStatusInput,
    category_names: &str,
    parse_category: F,
) -> Result<Option<T>, String>
where
    F: Fn(&str) -> Result<T, E>,
    E: std::fmt::Display,
{
    if status.category_missing_arg {
        return Err(super::usage::format_health_missing_category_arg_message(
            category_names,
        ));
    }

    if let Some(cat_str) = status.category.as_deref() {
        if cat_str == "all" {
            return Ok(None);
        }

        match parse_category(cat_str) {
            Ok(category) => Ok(Some(category)),
            Err(error) => Err(super::usage::format_health_invalid_category_message(
                &error.to_string(),
                category_names,
            )),
        }
    } else {
        Ok(None)
    }
}
