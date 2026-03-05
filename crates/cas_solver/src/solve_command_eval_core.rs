#[derive(Debug, Clone)]
pub struct PreparedSolveEvalRequest {
    pub request: crate::EvalRequest,
    pub var: String,
    pub original_equation: Option<cas_ast::Equation>,
}

#[derive(Debug, Clone)]
pub struct SolveCommandEvalOutput {
    pub var: String,
    pub original_equation: Option<cas_ast::Equation>,
    pub output: crate::EvalOutputView,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveCommandEvalError {
    Prepare(crate::SolvePrepareError),
    Eval(String),
}

pub fn prepare_solve_eval_request(
    ctx: &mut cas_ast::Context,
    input: &str,
    explicit_var: Option<String>,
    auto_store: bool,
) -> Result<PreparedSolveEvalRequest, crate::SolvePrepareError> {
    let (parsed_expr, original_equation, var) =
        crate::prepare_solve_expr_and_var(ctx, input, explicit_var)?;

    Ok(PreparedSolveEvalRequest {
        request: crate::EvalRequest {
            raw_input: input.to_string(),
            parsed: parsed_expr,
            action: crate::EvalAction::Solve { var: var.clone() },
            auto_store,
        },
        var,
        original_equation,
    })
}

pub fn evaluate_solve_command_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    parsed_input: crate::SolveCommandInput,
    auto_store: bool,
) -> Result<SolveCommandEvalOutput, SolveCommandEvalError>
where
    S: crate::EvalSession<Options = crate::EvalOptions, Diagnostics = crate::Diagnostics>,
    S::Store: crate::EvalStore<
        DomainMode = crate::DomainMode,
        RequiredItem = crate::RequiredItem,
        Step = crate::Step,
        Diagnostics = crate::Diagnostics,
    >,
{
    let PreparedSolveEvalRequest {
        request,
        var,
        original_equation,
    } = prepare_solve_eval_request(
        &mut engine.simplifier.context,
        parsed_input.equation.trim(),
        parsed_input.variable,
        auto_store,
    )
    .map_err(SolveCommandEvalError::Prepare)?;

    let output = engine
        .eval(session, request)
        .map_err(|e| SolveCommandEvalError::Eval(e.to_string()))?;
    let output = crate::eval_output_view(&output);

    Ok(SolveCommandEvalOutput {
        var,
        original_equation,
        output,
    })
}
