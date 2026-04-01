pub(super) fn wrap_expand_eval_expression(expr: &str) -> String {
    format!("expand({expr})")
}

pub(super) fn wrap_collect_eval_expression(expr: &str, var: &str) -> String {
    format!("collect({expr}, {var})")
}
