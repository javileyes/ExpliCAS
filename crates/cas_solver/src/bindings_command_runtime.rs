use crate::{
    binding_overview_entries, clear_bindings_command, evaluate_vars_command_lines,
    evaluate_vars_command_lines_with_context, format_clear_bindings_result_lines, BindingsContext,
};

/// Evaluate `clear` command and return output lines.
pub fn evaluate_clear_bindings_command_lines<C: BindingsContext>(
    context: &mut C,
    input: &str,
) -> Vec<String> {
    let result = clear_bindings_command(context, input);
    format_clear_bindings_result_lines(&result)
}

/// Evaluate `vars` command lines using an expression renderer callback.
pub fn evaluate_vars_command_lines_from_bindings<C: BindingsContext, F>(
    context: &C,
    render_expr: F,
) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
{
    let bindings = binding_overview_entries(context);
    evaluate_vars_command_lines(&bindings, render_expr)
}

/// Evaluate `vars` command lines using an explicit AST context.
pub fn evaluate_vars_command_lines_from_bindings_with_context<C: BindingsContext>(
    context: &C,
    ast_context: &cas_ast::Context,
) -> Vec<String> {
    let bindings = binding_overview_entries(context);
    evaluate_vars_command_lines_with_context(&bindings, ast_context)
}
