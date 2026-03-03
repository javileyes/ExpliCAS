use cas_ast::{Context, ExprId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisualizeEvalError {
    Parse(String),
}

#[derive(Debug, Clone)]
pub struct ExplainGcdEvalOutput {
    pub steps: Vec<String>,
    pub value: Option<ExprId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExplainCommandEvalError {
    Parse(String),
    ExpectedFunctionCall,
    UnsupportedFunction(String),
    InvalidArity {
        function: String,
        expected: usize,
        found: usize,
    },
}

pub fn evaluate_visualize_ast_dot(
    ctx: &mut Context,
    input: &str,
) -> Result<String, VisualizeEvalError> {
    let parsed_expr = cas_parser::parse(input.trim(), ctx)
        .map_err(|e| VisualizeEvalError::Parse(e.to_string()))?;
    let mut viz = crate::visualizer::AstVisualizer::new(ctx);
    Ok(viz.to_dot(parsed_expr))
}

pub fn evaluate_explain_gcd_command(
    ctx: &mut Context,
    input: &str,
) -> Result<ExplainGcdEvalOutput, ExplainCommandEvalError> {
    let parsed_expr = cas_parser::parse(input.trim(), ctx)
        .map_err(|e| ExplainCommandEvalError::Parse(e.to_string()))?;
    let expr_data = ctx.get(parsed_expr).clone();
    let cas_ast::Expr::Function(name_id, args) = expr_data else {
        return Err(ExplainCommandEvalError::ExpectedFunctionCall);
    };
    let function_name = ctx.sym_name(name_id).to_string();
    if function_name != "gcd" {
        return Err(ExplainCommandEvalError::UnsupportedFunction(function_name));
    }
    if args.len() != 2 {
        return Err(ExplainCommandEvalError::InvalidArity {
            function: function_name,
            expected: 2,
            found: args.len(),
        });
    }

    let result = crate::number_theory::explain_gcd(ctx, args[0], args[1]);
    Ok(ExplainGcdEvalOutput {
        steps: result.steps,
        value: result.value,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_explain_gcd_command, evaluate_visualize_ast_dot, ExplainCommandEvalError,
        VisualizeEvalError,
    };

    #[test]
    fn evaluate_visualize_ast_dot_returns_graph() {
        let mut ctx = cas_ast::Context::new();
        let dot = evaluate_visualize_ast_dot(&mut ctx, "x+1").expect("visualize");
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn evaluate_visualize_ast_dot_parse_error_is_typed() {
        let mut ctx = cas_ast::Context::new();
        let err = evaluate_visualize_ast_dot(&mut ctx, "x+").expect_err("parse error");
        assert!(matches!(err, VisualizeEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_explain_gcd_command_returns_steps() {
        let mut ctx = cas_ast::Context::new();
        let out = evaluate_explain_gcd_command(&mut ctx, "gcd(48, 18)").expect("explain gcd");
        assert!(!out.steps.is_empty());
        assert!(out.value.is_some());
    }

    #[test]
    fn evaluate_explain_gcd_command_rejects_non_function() {
        let mut ctx = cas_ast::Context::new();
        let err = evaluate_explain_gcd_command(&mut ctx, "x+1").expect_err("expected rejection");
        assert_eq!(err, ExplainCommandEvalError::ExpectedFunctionCall);
    }

    #[test]
    fn evaluate_explain_gcd_command_rejects_unsupported_function() {
        let mut ctx = cas_ast::Context::new();
        let err =
            evaluate_explain_gcd_command(&mut ctx, "lcm(4, 6)").expect_err("expected rejection");
        assert_eq!(
            err,
            ExplainCommandEvalError::UnsupportedFunction("lcm".to_string())
        );
    }
}
