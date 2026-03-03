use cas_ast::{Context, ExprId};

use crate::{Simplifier, Step, TelescopingResult};

#[derive(Debug, Clone)]
pub struct UnaryFunctionEvalOutput {
    pub result_expr: ExprId,
    pub steps: Vec<Step>,
}

pub struct TelescopeEvalOutput {
    pub parsed_expr: ExprId,
    pub result: TelescopingResult,
}

#[derive(Debug, Clone)]
pub struct ExpandLogEvalOutput {
    pub parsed_expr: ExprId,
    pub expanded_expr: ExprId,
}

#[derive(Debug, Clone)]
pub struct WeierstrassEvalOutput {
    pub parsed_expr: ExprId,
    pub substituted_expr: ExprId,
    pub simplified_expr: ExprId,
}

pub fn evaluate_unary_function_command(
    simplifier: &mut Simplifier,
    function_name: &str,
    input: &str,
) -> Result<UnaryFunctionEvalOutput, String> {
    let parsed_expr = cas_parser::parse(input.trim(), &mut simplifier.context)
        .map_err(|e| format!("Parse error: {e}"))?;
    let call_expr = simplifier.context.call(function_name, vec![parsed_expr]);
    let (result_expr, steps) = simplifier.simplify(call_expr);
    Ok(UnaryFunctionEvalOutput { result_expr, steps })
}

pub fn evaluate_telescope_command(
    ctx: &mut Context,
    input: &str,
) -> Result<TelescopeEvalOutput, String> {
    let parsed_expr =
        cas_parser::parse(input.trim(), ctx).map_err(|e| format!("Parse error: {e}"))?;
    let result = crate::telescope(ctx, parsed_expr);
    Ok(TelescopeEvalOutput {
        parsed_expr,
        result,
    })
}

pub fn evaluate_expand_log_command(
    ctx: &mut Context,
    input: &str,
) -> Result<ExpandLogEvalOutput, String> {
    let parsed_expr =
        cas_parser::parse(input.trim(), ctx).map_err(|e| format!("Parse error: {e}"))?;
    let expanded_expr = crate::expand_log_recursive(ctx, parsed_expr);
    Ok(ExpandLogEvalOutput {
        parsed_expr,
        expanded_expr,
    })
}

pub fn evaluate_weierstrass_command(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<WeierstrassEvalOutput, String> {
    let parsed_expr = cas_parser::parse(input.trim(), &mut simplifier.context)
        .map_err(|e| format!("Parse error: {e}"))?;
    let substituted_expr = crate::apply_weierstrass_recursive(&mut simplifier.context, parsed_expr);
    let (simplified_expr, _steps) = simplifier.simplify(substituted_expr);
    Ok(WeierstrassEvalOutput {
        parsed_expr,
        substituted_expr,
        simplified_expr,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_expand_log_command, evaluate_telescope_command, evaluate_unary_function_command,
        evaluate_weierstrass_command,
    };

    #[test]
    fn evaluate_unary_function_command_runs() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let out = evaluate_unary_function_command(&mut simplifier, "trace", "[[1,2],[3,4]]")
            .expect("unary eval");
        let shown = cas_formatter::render_expr(&simplifier.context, out.result_expr);
        assert_eq!(shown, "5");
    }

    #[test]
    fn evaluate_telescope_command_runs() {
        let mut ctx = cas_ast::Context::new();
        let out = evaluate_telescope_command(&mut ctx, "1 + 2*cos(x)").expect("telescope eval");
        let formatted = out.result.format(&ctx);
        assert!(!formatted.is_empty());
    }

    #[test]
    fn evaluate_expand_log_command_runs() {
        let mut ctx = cas_ast::Context::new();
        let out = evaluate_expand_log_command(&mut ctx, "ln(x^2*y)").expect("expand_log eval");
        let shown = cas_formatter::render_expr(&ctx, out.expanded_expr);
        assert!(shown.contains("ln"));
    }

    #[test]
    fn evaluate_weierstrass_command_runs() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let out = evaluate_weierstrass_command(&mut simplifier, "sin(x) + cos(x)")
            .expect("weierstrass eval");
        let shown = cas_formatter::render_expr(&simplifier.context, out.simplified_expr);
        assert!(!shown.is_empty());
    }
}
