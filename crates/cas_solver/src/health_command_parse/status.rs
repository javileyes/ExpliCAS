use cas_solver_core::health_runtime::HealthStatusInput;

pub(super) fn parse_health_status_input(parts: &[&str]) -> HealthStatusInput {
    let opts: Vec<&str> = parts.iter().skip(2).copied().collect();
    let list_only = opts.contains(&"--list") || opts.contains(&"-l");
    let mut category = None;
    let mut category_missing_arg = false;

    if let Some(idx) = opts.iter().position(|&x| x == "--category" || x == "-c") {
        if let Some(cat) = opts.get(idx + 1) {
            category = Some((*cat).to_string());
        } else {
            category_missing_arg = true;
        }
    }

    HealthStatusInput {
        list_only,
        category,
        category_missing_arg,
    }
}
