pub(super) fn wrap_expand_eval_expression(expr: &str) -> String {
    format!("expand({expr})")
}
