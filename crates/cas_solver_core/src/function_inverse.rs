use cas_ast::{
    symbol::SymbolId, BuiltinFn, Constant, Context, Equation, Expr, ExprId, RelOp, SolutionSet,
};

/// Supported unary function inversion kinds used by equation isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryInverseKind {
    Sqrt,
    Ln,
    Exp,
    Sin,
    Cos,
    Tan,
}

/// Routing decision for function isolation entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionIsolationRoute {
    AbsUnary { arg: ExprId },
    LogBinary { base: ExprId, arg: ExprId },
    UnaryInvertible { arg: ExprId },
}

/// Routing errors for function isolation entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionIsolationRouteError {
    VariableNotFoundInUnaryArg,
    UnsupportedArity,
}

/// Derive entry routing for function isolation.
pub fn derive_function_isolation_route(
    ctx: &Context,
    fn_id: SymbolId,
    args: &[ExprId],
    var: &str,
) -> Result<FunctionIsolationRoute, FunctionIsolationRouteError> {
    if ctx.is_builtin(fn_id, BuiltinFn::Abs) && args.len() == 1 {
        return Ok(FunctionIsolationRoute::AbsUnary { arg: args[0] });
    }

    if ctx.is_builtin(fn_id, BuiltinFn::Log) && args.len() == 2 {
        return Ok(FunctionIsolationRoute::LogBinary {
            base: args[0],
            arg: args[1],
        });
    }

    if args.len() == 1 {
        let arg = args[0];
        if super::isolation_utils::contains_var(ctx, arg, var) {
            Ok(FunctionIsolationRoute::UnaryInvertible { arg })
        } else {
            Err(FunctionIsolationRouteError::VariableNotFoundInUnaryArg)
        }
    } else {
        Err(FunctionIsolationRouteError::UnsupportedArity)
    }
}

/// Concrete rewrite plan for unary inverse isolation.
#[derive(Debug, Clone, PartialEq)]
pub struct UnaryInverseRewritePlan {
    pub equation: Equation,
    pub step_description: &'static str,
    pub needs_rhs_cleanup: bool,
}

/// Didactic payload for unary inverse rewrites.
#[derive(Debug, Clone, PartialEq)]
pub struct UnaryInverseDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// Combined unary-inverse rewrite plus didactic step for solver orchestration.
#[derive(Debug, Clone, PartialEq)]
pub struct UnaryInverseIsolationStepPlan {
    pub equation: Equation,
    pub items: Vec<UnaryInverseExecutionItem>,
    pub needs_rhs_cleanup: bool,
}

/// One executable unary-inverse item aligned with didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct UnaryInverseExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl UnaryInverseExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Collect unary-inverse didactic steps in display order.
pub fn collect_unary_inverse_didactic_steps(
    plan: &UnaryInverseIsolationStepPlan,
) -> Vec<UnaryInverseDidacticStep> {
    plan.items
        .iter()
        .cloned()
        .map(|item| UnaryInverseDidacticStep {
            description: item.description,
            equation_after: item.equation,
        })
        .collect()
}

/// Collect unary-inverse execution items in display order.
pub fn collect_unary_inverse_execution_items(
    plan: &UnaryInverseIsolationStepPlan,
) -> Vec<UnaryInverseExecutionItem> {
    plan.items.clone()
}

/// Didactic payload for RHS cleanup steps emitted after inverse rewrites.
#[derive(Debug, Clone, PartialEq)]
pub struct RhsSimplificationDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// One executable RHS-simplification item aligned with didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct RhsSimplificationExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl RhsSimplificationExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Engine-facing unary-inverse execution payload.
#[derive(Debug, Clone, PartialEq)]
pub struct UnaryInverseSolveExecution {
    pub rewrite_items: Vec<UnaryInverseExecutionItem>,
    pub rhs_cleanup_items: Vec<RhsSimplificationExecutionItem>,
    pub rewritten_equation: Equation,
    pub target_rhs: ExprId,
}

/// One executable unary-inverse solve item (rewrite or RHS cleanup).
#[derive(Debug, Clone, PartialEq)]
pub struct UnaryInverseSolveExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl UnaryInverseSolveExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Solved payload for one unary-inverse execution.
#[derive(Debug, Clone, PartialEq)]
pub struct UnaryInverseSolvedExecution<T> {
    pub execution: UnaryInverseSolveExecution,
    pub solved: T,
}

/// Collect unary-inverse solve execution items in display order:
/// rewrite items first, RHS-cleanup items second.
pub fn collect_unary_inverse_solve_execution_items(
    execution: &UnaryInverseSolveExecution,
) -> Vec<UnaryInverseSolveExecutionItem> {
    execution
        .rewrite_items
        .iter()
        .map(|item| UnaryInverseSolveExecutionItem {
            equation: item.equation.clone(),
            description: item.description.clone(),
        })
        .chain(
            execution
                .rhs_cleanup_items
                .iter()
                .map(|item| UnaryInverseSolveExecutionItem {
                    equation: item.equation.clone(),
                    description: item.description.clone(),
                }),
        )
        .collect()
}

impl UnaryInverseKind {
    /// Classify a unary function name into an inversion kind.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "sqrt" => Some(Self::Sqrt),
            "ln" => Some(Self::Ln),
            "exp" => Some(Self::Exp),
            "sin" => Some(Self::Sin),
            "cos" => Some(Self::Cos),
            "tan" => Some(Self::Tan),
            _ => None,
        }
    }

    /// Build the transformed RHS when inverting `f(arg) = rhs`.
    pub fn build_rhs(self, ctx: &mut Context, rhs: ExprId) -> ExprId {
        match self {
            Self::Sqrt => {
                let two = ctx.num(2);
                ctx.add(Expr::Pow(rhs, two))
            }
            Self::Ln => {
                let e = ctx.add(Expr::Constant(Constant::E));
                ctx.add(Expr::Pow(e, rhs))
            }
            Self::Exp => ctx.call("ln", vec![rhs]),
            Self::Sin => ctx.call("arcsin", vec![rhs]),
            Self::Cos => ctx.call("arccos", vec![rhs]),
            Self::Tan => ctx.call("arctan", vec![rhs]),
        }
    }

    /// Whether the inverted RHS should be simplified before recursive isolation.
    ///
    /// Inverse trig functions often produce rewrite-friendly forms.
    pub fn needs_rhs_cleanup(self) -> bool {
        matches!(self, Self::Sin | Self::Cos | Self::Tan)
    }

    /// Human-readable action used by solver step narration.
    pub fn step_description(self) -> &'static str {
        match self {
            Self::Ln => "Exponentiate both sides with base e",
            Self::Exp => "Take natural log of both sides",
            Self::Sqrt => "Square both sides",
            Self::Sin => "Take arcsin of both sides",
            Self::Cos => "Take arccos of both sides",
            Self::Tan => "Take arctan of both sides",
        }
    }

    /// Whether this inverse is allowed in `UnwrapStrategy`.
    pub fn supports_unwrap_strategy(self) -> bool {
        matches!(self, Self::Sqrt | Self::Ln | Self::Exp)
    }

    /// Step text used by `UnwrapStrategy`.
    pub fn unwrap_step_description(self) -> Option<&'static str> {
        match self {
            Self::Sqrt => Some("Square both sides"),
            Self::Ln => Some("Exponentiate (base e)"),
            Self::Exp => Some("Take natural log"),
            _ => None,
        }
    }
}

/// Rewrite a unary-function equation by applying its inverse:
/// `f(arg) op other` -> `arg op inverse_f(other)` (when `is_lhs=true`),
/// or symmetrically when the function is on RHS.
pub fn rewrite_unary_inverse_equation(
    ctx: &mut Context,
    fn_name: &str,
    arg: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
) -> Option<(Equation, UnaryInverseKind)> {
    let inverse_kind = UnaryInverseKind::from_name(fn_name)?;
    let transformed_other = inverse_kind.build_rhs(ctx, other);
    let equation = if is_lhs {
        Equation {
            lhs: arg,
            rhs: transformed_other,
            op,
        }
    } else {
        Equation {
            lhs: transformed_other,
            rhs: arg,
            op,
        }
    };
    Some((equation, inverse_kind))
}

/// Same as [`rewrite_unary_inverse_equation`] but restricted to unwrap-safe inverses.
pub fn rewrite_unary_inverse_equation_for_unwrap(
    ctx: &mut Context,
    fn_name: &str,
    arg: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
) -> Option<(Equation, UnaryInverseKind)> {
    let (equation, inverse_kind) =
        rewrite_unary_inverse_equation(ctx, fn_name, arg, other, op, is_lhs)?;
    if inverse_kind.supports_unwrap_strategy() {
        Some((equation, inverse_kind))
    } else {
        None
    }
}

/// Build an unwrap-safe unary inverse rewrite plan.
///
/// Returns `None` when the function is unsupported by `UnwrapStrategy`.
pub fn plan_unary_inverse_rewrite_for_unwrap(
    ctx: &mut Context,
    fn_name: &str,
    arg: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
) -> Option<(Equation, &'static str)> {
    let (equation, inverse_kind) =
        rewrite_unary_inverse_equation_for_unwrap(ctx, fn_name, arg, other, op, is_lhs)?;
    let description = inverse_kind
        .unwrap_step_description()
        .expect("unwrap filter guarantees supported inverse description");
    Some((equation, description))
}

/// Build a full unary inverse rewrite plan with narration metadata.
pub fn plan_unary_inverse_rewrite(
    ctx: &mut Context,
    fn_name: &str,
    arg: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
) -> Option<UnaryInverseRewritePlan> {
    let (equation, inverse_kind) =
        rewrite_unary_inverse_equation(ctx, fn_name, arg, other, op, is_lhs)?;
    Some(UnaryInverseRewritePlan {
        equation,
        step_description: inverse_kind.step_description(),
        needs_rhs_cleanup: inverse_kind.needs_rhs_cleanup(),
    })
}

/// Build didactic payload from a unary inverse rewrite plan.
pub fn build_unary_inverse_step(plan: &UnaryInverseRewritePlan) -> UnaryInverseDidacticStep {
    UnaryInverseDidacticStep {
        description: plan.step_description.to_string(),
        equation_after: plan.equation.clone(),
    }
}

/// Plan unary inverse isolation with prebuilt didactic payload.
pub fn plan_unary_inverse_isolation_step(
    ctx: &mut Context,
    fn_name: &str,
    arg: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
) -> Option<UnaryInverseIsolationStepPlan> {
    let rewrite = plan_unary_inverse_rewrite(ctx, fn_name, arg, other, op, is_lhs)?;
    let items = vec![UnaryInverseExecutionItem {
        equation: rewrite.equation.clone(),
        description: rewrite.step_description.to_string(),
    }];
    Some(UnaryInverseIsolationStepPlan {
        equation: rewrite.equation,
        items,
        needs_rhs_cleanup: rewrite.needs_rhs_cleanup,
    })
}

/// Execute unary-inverse isolation with optional RHS cleanup using closure hooks.
///
/// Returns `None` when function inversion is unsupported.
pub fn execute_unary_inverse_with<FPlan, FSimplifyRhs>(
    fn_name: &str,
    arg: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
    mut plan_unary_inverse_step: FPlan,
    mut simplify_rhs_with_entries: FSimplifyRhs,
) -> Option<UnaryInverseSolveExecution>
where
    FPlan: FnMut(&str, ExprId, ExprId, RelOp, bool) -> Option<UnaryInverseIsolationStepPlan>,
    FSimplifyRhs: FnMut(ExprId) -> (ExprId, Vec<(String, ExprId)>),
{
    let plan = plan_unary_inverse_step(fn_name, arg, other, op, is_lhs)?;
    let rewritten_equation = plan.equation.clone();
    let mut rhs_cleanup_items = Vec::new();
    let mut target_rhs = rewritten_equation.rhs;

    if plan.needs_rhs_cleanup {
        let (simplified_rhs, entries) = simplify_rhs_with_entries(target_rhs);
        target_rhs = simplified_rhs;
        rhs_cleanup_items = build_rhs_simplification_execution_items(
            rewritten_equation.lhs,
            rewritten_equation.op.clone(),
            entries,
        );
    }

    Some(UnaryInverseSolveExecution {
        rewrite_items: plan.items,
        rhs_cleanup_items,
        rewritten_equation,
        target_rhs,
    })
}

/// Execute recursive solve for unary-inverse rewrite output.
pub fn solve_unary_inverse_execution_with<E, T, FSolve>(
    execution: UnaryInverseSolveExecution,
    mut solve: FSolve,
) -> Result<UnaryInverseSolvedExecution<T>, E>
where
    FSolve: FnMut(ExprId, ExprId, RelOp) -> Result<T, E>,
{
    let solved = solve(
        execution.rewritten_equation.lhs,
        execution.target_rhs,
        execution.rewritten_equation.op.clone(),
    )?;
    Ok(UnaryInverseSolvedExecution { execution, solved })
}

/// Execute recursive solve for unary-inverse rewrite output while passing
/// aligned solve execution items to the solve callback.
pub fn solve_unary_inverse_execution_with_items<E, T, FSolve>(
    execution: UnaryInverseSolveExecution,
    mut solve: FSolve,
) -> Result<UnaryInverseSolvedExecution<T>, E>
where
    FSolve: FnMut(Vec<UnaryInverseSolveExecutionItem>, ExprId, ExprId, RelOp) -> Result<T, E>,
{
    let items = collect_unary_inverse_solve_execution_items(&execution);
    let solved = solve(
        items,
        execution.rewritten_equation.lhs,
        execution.target_rhs,
        execution.rewritten_equation.op.clone(),
    )?;
    Ok(UnaryInverseSolvedExecution { execution, solved })
}

/// Solved result for unary-inverse execution pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct UnaryInverseExecutionPipelineSolved<S> {
    pub solution_set: SolutionSet,
    pub steps: Vec<S>,
}

/// Execute unary-inverse solve + optional didactic item dispatch.
pub fn solve_unary_inverse_execution_pipeline_with_items<E, S, FSolve, FStep>(
    execution: UnaryInverseSolveExecution,
    include_items: bool,
    mut solve: FSolve,
    mut map_item_to_step: FStep,
) -> Result<UnaryInverseExecutionPipelineSolved<S>, E>
where
    FSolve: FnMut(ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(UnaryInverseSolveExecutionItem) -> S,
{
    let lhs = execution.rewritten_equation.lhs;
    let rhs = execution.target_rhs;
    let op = execution.rewritten_equation.op.clone();
    let mut steps = Vec::new();
    if include_items {
        for item in collect_unary_inverse_solve_execution_items(&execution) {
            steps.push(map_item_to_step(item));
        }
    }
    let (solution_set, mut sub_steps) = solve(lhs, rhs, op)?;
    steps.append(&mut sub_steps);
    Ok(UnaryInverseExecutionPipelineSolved {
        solution_set,
        steps,
    })
}

/// Execute unary-inverse planning + solve pipeline with optional didactic item dispatch.
///
/// Returns `None` when unary inversion is unsupported for the requested function.
#[allow(clippy::too_many_arguments)]
pub fn execute_unary_inverse_pipeline_with_items_with<E, S, FPlan, FSimplifyRhs, FSolve, FStep>(
    fn_name: &str,
    arg: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
    include_items: bool,
    plan_unary_inverse_step: FPlan,
    simplify_rhs_with_entries: FSimplifyRhs,
    solve: FSolve,
    map_item_to_step: FStep,
) -> Option<Result<UnaryInverseExecutionPipelineSolved<S>, E>>
where
    FPlan: FnMut(&str, ExprId, ExprId, RelOp, bool) -> Option<UnaryInverseIsolationStepPlan>,
    FSimplifyRhs: FnMut(ExprId) -> (ExprId, Vec<(String, ExprId)>),
    FSolve: FnMut(ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(UnaryInverseSolveExecutionItem) -> S,
{
    let execution = execute_unary_inverse_with(
        fn_name,
        arg,
        other,
        op,
        is_lhs,
        plan_unary_inverse_step,
        simplify_rhs_with_entries,
    )?;
    Some(solve_unary_inverse_execution_pipeline_with_items(
        execution,
        include_items,
        solve,
        map_item_to_step,
    ))
}

/// Execute unary-inverse planning + solve pipeline returning plain strategy
/// output `(SolutionSet, steps)`.
///
/// Returns `None` when unary inversion is unsupported for the requested function.
#[allow(clippy::too_many_arguments)]
pub fn execute_unary_inverse_result_pipeline_with_items_with<
    E,
    S,
    FPlan,
    FSimplifyRhs,
    FSolve,
    FStep,
>(
    fn_name: &str,
    arg: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
    include_items: bool,
    plan_unary_inverse_step: FPlan,
    simplify_rhs_with_entries: FSimplifyRhs,
    solve: FSolve,
    map_item_to_step: FStep,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FPlan: FnMut(&str, ExprId, ExprId, RelOp, bool) -> Option<UnaryInverseIsolationStepPlan>,
    FSimplifyRhs: FnMut(ExprId) -> (ExprId, Vec<(String, ExprId)>),
    FSolve: FnMut(ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(UnaryInverseSolveExecutionItem) -> S,
{
    let solved = execute_unary_inverse_pipeline_with_items_with(
        fn_name,
        arg,
        other,
        op,
        is_lhs,
        include_items,
        plan_unary_inverse_step,
        simplify_rhs_with_entries,
        solve,
        map_item_to_step,
    )?;
    Some(solved.map(|payload| (payload.solution_set, payload.steps)))
}

/// Build didactic RHS-cleanup steps from `(description, rhs_after)` tuples.
pub fn build_rhs_simplification_steps<I>(
    lhs: ExprId,
    op: RelOp,
    entries: I,
) -> Vec<RhsSimplificationDidacticStep>
where
    I: IntoIterator<Item = (String, ExprId)>,
{
    entries
        .into_iter()
        .map(|(description, rhs_after)| RhsSimplificationDidacticStep {
            description,
            equation_after: Equation {
                lhs,
                rhs: rhs_after,
                op: op.clone(),
            },
        })
        .collect()
}

/// Collect RHS-simplification execution items in display order.
pub fn collect_rhs_simplification_execution_items(
    steps: &[RhsSimplificationDidacticStep],
) -> Vec<RhsSimplificationExecutionItem> {
    steps
        .iter()
        .cloned()
        .map(|didactic| RhsSimplificationExecutionItem {
            equation: didactic.equation_after.clone(),
            description: didactic.description,
        })
        .collect()
}

/// Build RHS-simplification execution items directly from `(description, rhs_after)`
/// tuples, fixing `lhs`/`op` for each generated equation.
pub fn build_rhs_simplification_execution_items<I>(
    lhs: ExprId,
    op: RelOp,
    entries: I,
) -> Vec<RhsSimplificationExecutionItem>
where
    I: IntoIterator<Item = (String, ExprId)>,
{
    entries
        .into_iter()
        .map(|(description, rhs_after)| RhsSimplificationExecutionItem {
            equation: Equation {
                lhs,
                rhs: rhs_after,
                op: op.clone(),
            },
            description,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::SolutionSet;

    #[test]
    fn classify_supported_names() {
        assert_eq!(
            UnaryInverseKind::from_name("sqrt"),
            Some(UnaryInverseKind::Sqrt)
        );
        assert_eq!(
            UnaryInverseKind::from_name("ln"),
            Some(UnaryInverseKind::Ln)
        );
        assert_eq!(
            UnaryInverseKind::from_name("exp"),
            Some(UnaryInverseKind::Exp)
        );
        assert_eq!(
            UnaryInverseKind::from_name("sin"),
            Some(UnaryInverseKind::Sin)
        );
        assert_eq!(
            UnaryInverseKind::from_name("cos"),
            Some(UnaryInverseKind::Cos)
        );
        assert_eq!(
            UnaryInverseKind::from_name("tan"),
            Some(UnaryInverseKind::Tan)
        );
        assert_eq!(UnaryInverseKind::from_name("unknown"), None);
    }

    #[test]
    fn derive_function_isolation_route_abs_unary() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let abs = ctx.call("abs", vec![x]);
        let (fn_id, args) = match ctx.get(abs) {
            Expr::Function(fn_id, args) => (*fn_id, args.clone()),
            other => panic!("expected function node, got {:?}", other),
        };

        let route = derive_function_isolation_route(&ctx, fn_id, &args, "x")
            .expect("abs unary route should be supported");
        assert_eq!(route, FunctionIsolationRoute::AbsUnary { arg: x });
    }

    #[test]
    fn derive_function_isolation_route_log_binary() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let x = ctx.var("x");
        let log = ctx.call("log", vec![b, x]);
        let (fn_id, args) = match ctx.get(log) {
            Expr::Function(fn_id, args) => (*fn_id, args.clone()),
            other => panic!("expected function node, got {:?}", other),
        };

        let route = derive_function_isolation_route(&ctx, fn_id, &args, "x")
            .expect("log binary route should be supported");
        assert_eq!(route, FunctionIsolationRoute::LogBinary { base: b, arg: x });
    }

    #[test]
    fn derive_function_isolation_route_unary_invertible_when_arg_has_variable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sin = ctx.call("sin", vec![x]);
        let (fn_id, args) = match ctx.get(sin) {
            Expr::Function(fn_id, args) => (*fn_id, args.clone()),
            other => panic!("expected function node, got {:?}", other),
        };

        let route = derive_function_isolation_route(&ctx, fn_id, &args, "x")
            .expect("unary invertible route should be selected");
        assert_eq!(route, FunctionIsolationRoute::UnaryInvertible { arg: x });
    }

    #[test]
    fn derive_function_isolation_route_errors_when_unary_arg_lacks_variable() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let sin = ctx.call("sin", vec![y]);
        let (fn_id, args) = match ctx.get(sin) {
            Expr::Function(fn_id, args) => (*fn_id, args.clone()),
            other => panic!("expected function node, got {:?}", other),
        };

        let route = derive_function_isolation_route(&ctx, fn_id, &args, "x");
        assert_eq!(
            route,
            Err(FunctionIsolationRouteError::VariableNotFoundInUnaryArg)
        );
    }

    #[test]
    fn derive_function_isolation_route_errors_on_unsupported_arity() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let f = ctx.call("f", vec![x, y, z]);
        let (fn_id, args) = match ctx.get(f) {
            Expr::Function(fn_id, args) => (*fn_id, args.clone()),
            other => panic!("expected function node, got {:?}", other),
        };

        let route = derive_function_isolation_route(&ctx, fn_id, &args, "x");
        assert_eq!(route, Err(FunctionIsolationRouteError::UnsupportedArity));
    }

    #[test]
    fn build_ln_inverse_rhs_is_e_pow_rhs() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let rhs = UnaryInverseKind::Ln.build_rhs(&mut ctx, y);
        match ctx.get(rhs) {
            Expr::Pow(base, exp) => {
                assert_eq!(*exp, y);
                assert!(matches!(ctx.get(*base), Expr::Constant(Constant::E)));
            }
            other => panic!("Expected Pow(e, y), got {:?}", other),
        }
    }

    #[test]
    fn trig_inverses_require_cleanup() {
        assert!(UnaryInverseKind::Sin.needs_rhs_cleanup());
        assert!(UnaryInverseKind::Cos.needs_rhs_cleanup());
        assert!(UnaryInverseKind::Tan.needs_rhs_cleanup());
        assert!(!UnaryInverseKind::Ln.needs_rhs_cleanup());
        assert!(!UnaryInverseKind::Exp.needs_rhs_cleanup());
        assert!(!UnaryInverseKind::Sqrt.needs_rhs_cleanup());
    }

    #[test]
    fn step_descriptions_match_expected_wording() {
        assert_eq!(
            UnaryInverseKind::Ln.step_description(),
            "Exponentiate both sides with base e"
        );
        assert_eq!(
            UnaryInverseKind::Exp.step_description(),
            "Take natural log of both sides"
        );
        assert_eq!(
            UnaryInverseKind::Sqrt.step_description(),
            "Square both sides"
        );
        assert_eq!(
            UnaryInverseKind::Sin.step_description(),
            "Take arcsin of both sides"
        );
        assert_eq!(
            UnaryInverseKind::Cos.step_description(),
            "Take arccos of both sides"
        );
        assert_eq!(
            UnaryInverseKind::Tan.step_description(),
            "Take arctan of both sides"
        );
    }

    #[test]
    fn unwrap_strategy_support_is_restricted_to_ln_exp_sqrt() {
        assert!(UnaryInverseKind::Ln.supports_unwrap_strategy());
        assert!(UnaryInverseKind::Exp.supports_unwrap_strategy());
        assert!(UnaryInverseKind::Sqrt.supports_unwrap_strategy());
        assert!(!UnaryInverseKind::Sin.supports_unwrap_strategy());
        assert!(!UnaryInverseKind::Cos.supports_unwrap_strategy());
        assert!(!UnaryInverseKind::Tan.supports_unwrap_strategy());
    }

    #[test]
    fn unwrap_step_descriptions_match_expected_wording() {
        assert_eq!(
            UnaryInverseKind::Ln.unwrap_step_description(),
            Some("Exponentiate (base e)")
        );
        assert_eq!(
            UnaryInverseKind::Exp.unwrap_step_description(),
            Some("Take natural log")
        );
        assert_eq!(
            UnaryInverseKind::Sqrt.unwrap_step_description(),
            Some("Square both sides")
        );
        assert_eq!(UnaryInverseKind::Sin.unwrap_step_description(), None);
        assert_eq!(UnaryInverseKind::Cos.unwrap_step_description(), None);
        assert_eq!(UnaryInverseKind::Tan.unwrap_step_description(), None);
    }

    #[test]
    fn rewrite_unary_inverse_equation_lhs_orientation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let (eq, kind) = rewrite_unary_inverse_equation(&mut ctx, "sqrt", x, y, RelOp::Eq, true)
            .expect("sqrt inverse should rewrite");
        assert_eq!(kind, UnaryInverseKind::Sqrt);
        assert_eq!(eq.lhs, x);
        assert!(matches!(ctx.get(eq.rhs), Expr::Pow(base, _) if *base == y));
    }

    #[test]
    fn rewrite_unary_inverse_equation_rhs_orientation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let (eq, kind) = rewrite_unary_inverse_equation(&mut ctx, "exp", x, y, RelOp::Eq, false)
            .expect("exp inverse should rewrite");
        assert_eq!(kind, UnaryInverseKind::Exp);
        assert_eq!(eq.rhs, x);
        assert!(
            matches!(ctx.get(eq.lhs), Expr::Function(_, args) if args.len() == 1 && args[0] == y)
        );
    }

    #[test]
    fn rewrite_unary_inverse_equation_for_unwrap_rejects_trig() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        assert!(
            rewrite_unary_inverse_equation_for_unwrap(&mut ctx, "sin", x, y, RelOp::Eq, true)
                .is_none()
        );
    }

    #[test]
    fn plan_unary_inverse_rewrite_carries_step_metadata() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let plan = plan_unary_inverse_rewrite(&mut ctx, "exp", x, y, RelOp::Eq, true)
            .expect("exp inverse should build plan");
        assert_eq!(plan.step_description, "Take natural log of both sides");
        assert!(!plan.needs_rhs_cleanup);
        assert_eq!(plan.equation.lhs, x);
    }

    #[test]
    fn build_unary_inverse_step_uses_plan_description_and_equation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let plan = plan_unary_inverse_rewrite(&mut ctx, "sqrt", x, y, RelOp::Eq, true)
            .expect("sqrt should build plan");
        let step = build_unary_inverse_step(&plan);
        assert_eq!(step.description, "Square both sides");
        assert_eq!(step.equation_after, plan.equation);
    }

    #[test]
    fn plan_unary_inverse_isolation_step_builds_rewrite_and_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");

        let plan = plan_unary_inverse_isolation_step(&mut ctx, "sin", x, y, RelOp::Eq, true)
            .expect("sin inverse should build isolation plan");

        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.items[0].description, "Take arcsin of both sides");
        assert_eq!(plan.items[0].equation, plan.equation);
        assert!(plan.needs_rhs_cleanup);
    }

    #[test]
    fn collect_unary_inverse_didactic_steps_returns_single_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let plan = plan_unary_inverse_isolation_step(&mut ctx, "sin", x, y, RelOp::Eq, true)
            .expect("sin inverse should build isolation plan");

        let didactic = collect_unary_inverse_didactic_steps(&plan);
        assert_eq!(didactic.len(), 1);
        assert_eq!(didactic[0].description, plan.items[0].description);
        assert_eq!(didactic[0].equation_after, plan.items[0].equation);
    }

    #[test]
    fn collect_unary_inverse_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let plan = plan_unary_inverse_isolation_step(&mut ctx, "sin", x, y, RelOp::Eq, true)
            .expect("sin inverse should build isolation plan");

        let items = collect_unary_inverse_execution_items(&plan);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, plan.equation);
        assert_eq!(items[0].description, plan.items[0].description);
    }

    #[test]
    fn plan_unary_inverse_rewrite_for_unwrap_returns_description() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let (eq, desc) =
            plan_unary_inverse_rewrite_for_unwrap(&mut ctx, "ln", x, y, RelOp::Eq, true)
                .expect("ln should be unwrap-safe");
        assert_eq!(desc, "Exponentiate (base e)");
        assert_eq!(eq.lhs, x);
    }

    #[test]
    fn plan_unary_inverse_rewrite_for_unwrap_rejects_trig() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        assert!(
            plan_unary_inverse_rewrite_for_unwrap(&mut ctx, "sin", x, y, RelOp::Eq, true).is_none()
        );
    }

    #[test]
    fn build_rhs_simplification_steps_builds_equations_with_fixed_lhs() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");

        let out = build_rhs_simplification_steps(
            x,
            RelOp::Eq,
            vec![("step-1".to_string(), y), ("step-2".to_string(), z)],
        );

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].description, "step-1");
        assert_eq!(out[0].equation_after.lhs, x);
        assert_eq!(out[0].equation_after.rhs, y);
        assert_eq!(out[1].description, "step-2");
        assert_eq!(out[1].equation_after.lhs, x);
        assert_eq!(out[1].equation_after.rhs, z);
        assert_eq!(out[1].equation_after.op, RelOp::Eq);
    }

    #[test]
    fn collect_rhs_simplification_execution_items_preserves_order() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");

        let out = build_rhs_simplification_steps(
            x,
            RelOp::Eq,
            vec![("step-1".to_string(), y), ("step-2".to_string(), z)],
        );
        let items = collect_rhs_simplification_execution_items(&out);

        assert_eq!(items.len(), 2);
        assert_eq!(items[0].description, "step-1");
        assert_eq!(items[0].equation, out[0].equation_after);
        assert_eq!(items[1].description, "step-2");
        assert_eq!(items[1].equation, out[1].equation_after);
    }

    #[test]
    fn build_rhs_simplification_execution_items_builds_and_collects_in_one_pass() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");

        let items = build_rhs_simplification_execution_items(
            x,
            RelOp::Eq,
            vec![("step-1".to_string(), y), ("step-2".to_string(), z)],
        );

        assert_eq!(items.len(), 2);
        assert_eq!(items[0].description, "step-1");
        assert_eq!(items[0].equation.lhs, x);
        assert_eq!(items[0].equation.rhs, y);
        assert_eq!(items[1].description, "step-2");
        assert_eq!(items[1].equation.lhs, x);
        assert_eq!(items[1].equation.rhs, z);
    }

    #[test]
    fn execute_unary_inverse_with_applies_rhs_cleanup_when_needed() {
        let mut context = Context::new();
        let x = context.var("x");
        let y = context.var("y");
        let cleaned = context.var("cleaned");
        let mut simplify_calls = 0usize;

        let execution = execute_unary_inverse_with(
            "sin",
            x,
            y,
            RelOp::Eq,
            true,
            |fn_name, arg, other, op, is_lhs| {
                plan_unary_inverse_isolation_step(&mut context, fn_name, arg, other, op, is_lhs)
            },
            |rhs| {
                simplify_calls += 1;
                (
                    cleaned,
                    vec![
                        ("Simplify RHS".to_string(), rhs),
                        ("Done".to_string(), cleaned),
                    ],
                )
            },
        )
        .expect("sin inverse should execute");

        assert_eq!(simplify_calls, 1);
        assert_eq!(execution.rewrite_items.len(), 1);
        assert_eq!(execution.rhs_cleanup_items.len(), 2);
        assert_eq!(execution.target_rhs, cleaned);
    }

    #[test]
    fn solve_unary_inverse_execution_with_invokes_solver_once() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let execution = UnaryInverseSolveExecution {
            rewrite_items: vec![],
            rhs_cleanup_items: vec![],
            rewritten_equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            target_rhs: y,
        };

        let mut calls = 0usize;
        let solved = solve_unary_inverse_execution_with(execution, |lhs, rhs, op| {
            calls += 1;
            Ok::<_, ()>((lhs, rhs, op))
        })
        .expect("solve should succeed");

        assert_eq!(calls, 1);
        assert_eq!(solved.solved.0, x);
        assert_eq!(solved.solved.1, y);
        assert_eq!(solved.solved.2, RelOp::Eq);
    }

    #[test]
    fn solve_unary_inverse_execution_with_preserves_execution_payload() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let execution = UnaryInverseSolveExecution {
            rewrite_items: vec![UnaryInverseExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: y,
                    op: RelOp::Eq,
                },
                description: "rewrite".to_string(),
            }],
            rhs_cleanup_items: vec![RhsSimplificationExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: y,
                    op: RelOp::Eq,
                },
                description: "cleanup".to_string(),
            }],
            rewritten_equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            target_rhs: y,
        };
        let expected = execution.clone();

        let solved = solve_unary_inverse_execution_with(execution, |_lhs, _rhs, _op| {
            Ok::<_, ()>(SolutionSet::AllReals)
        })
        .expect("solve should succeed");

        assert_eq!(solved.execution, expected);
        assert!(matches!(solved.solved, SolutionSet::AllReals));
    }

    #[test]
    fn solve_unary_inverse_execution_with_items_passes_items_and_equation_parts() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let execution = UnaryInverseSolveExecution {
            rewrite_items: vec![UnaryInverseExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: y,
                    op: RelOp::Eq,
                },
                description: "rewrite".to_string(),
            }],
            rhs_cleanup_items: vec![RhsSimplificationExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: z,
                    op: RelOp::Eq,
                },
                description: "cleanup".to_string(),
            }],
            rewritten_equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            target_rhs: z,
        };

        let solved = solve_unary_inverse_execution_with_items(execution, |items, lhs, rhs, op| {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].description(), "rewrite");
            assert_eq!(items[1].description(), "cleanup");
            assert_eq!(lhs, x);
            assert_eq!(rhs, z);
            assert_eq!(op, RelOp::Eq);
            Ok::<_, ()>("ok")
        })
        .expect("solve should succeed");

        assert_eq!(solved.solved, "ok");
    }

    #[test]
    fn solve_unary_inverse_execution_pipeline_with_items_prepends_mapped_items() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let execution = UnaryInverseSolveExecution {
            rewrite_items: vec![UnaryInverseExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: y,
                    op: RelOp::Eq,
                },
                description: "rewrite".to_string(),
            }],
            rhs_cleanup_items: vec![RhsSimplificationExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: z,
                    op: RelOp::Eq,
                },
                description: "cleanup".to_string(),
            }],
            rewritten_equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            target_rhs: z,
        };

        let mut calls = 0usize;
        let solved = solve_unary_inverse_execution_pipeline_with_items(
            execution,
            true,
            |lhs, rhs, op| {
                calls += 1;
                assert_eq!(lhs, x);
                assert_eq!(rhs, z);
                assert_eq!(op, RelOp::Eq);
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![rhs]),
                    vec!["substep".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert_eq!(calls, 1);
        assert!(matches!(solved.solution_set, SolutionSet::Discrete(_)));
        assert_eq!(
            solved.steps,
            vec![
                "rewrite".to_string(),
                "cleanup".to_string(),
                "substep".to_string()
            ]
        );
    }

    #[test]
    fn solve_unary_inverse_execution_pipeline_with_items_omits_items_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let execution = UnaryInverseSolveExecution {
            rewrite_items: vec![UnaryInverseExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: y,
                    op: RelOp::Eq,
                },
                description: "rewrite".to_string(),
            }],
            rhs_cleanup_items: vec![],
            rewritten_equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            target_rhs: y,
        };

        let solved = solve_unary_inverse_execution_pipeline_with_items(
            execution,
            false,
            |_lhs, rhs, _op| {
                Ok::<_, ()>((SolutionSet::Discrete(vec![rhs]), vec!["only".to_string()]))
            },
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert!(matches!(solved.solution_set, SolutionSet::Discrete(_)));
        assert_eq!(solved.steps, vec!["only".to_string()]);
    }

    #[test]
    fn execute_unary_inverse_pipeline_with_items_with_runs_pipeline_for_supported_inverse() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");

        let mut solve_calls = 0usize;
        let solved = execute_unary_inverse_pipeline_with_items_with(
            "ln",
            x,
            y,
            RelOp::Eq,
            true,
            true,
            |_fn_name, _arg, _other, _op, _is_lhs| {
                Some(UnaryInverseIsolationStepPlan {
                    equation: Equation {
                        lhs: x,
                        rhs: y,
                        op: RelOp::Eq,
                    },
                    items: vec![UnaryInverseExecutionItem {
                        equation: Equation {
                            lhs: x,
                            rhs: y,
                            op: RelOp::Eq,
                        },
                        description: "rewrite".to_string(),
                    }],
                    needs_rhs_cleanup: true,
                })
            },
            |_rhs| (z, vec![("cleanup".to_string(), z)]),
            |lhs, rhs, op| {
                solve_calls += 1;
                assert_eq!(lhs, x);
                assert_eq!(rhs, z);
                assert_eq!(op, RelOp::Eq);
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![rhs]),
                    vec!["substep".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("inverse should be supported")
        .expect("pipeline should solve");

        assert_eq!(solve_calls, 1);
        assert!(matches!(solved.solution_set, SolutionSet::Discrete(_)));
        assert_eq!(
            solved.steps,
            vec![
                "rewrite".to_string(),
                "cleanup".to_string(),
                "substep".to_string()
            ]
        );
    }

    #[test]
    fn execute_unary_inverse_result_pipeline_with_items_with_returns_plain_tuple() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");

        let solved = execute_unary_inverse_result_pipeline_with_items_with(
            "ln",
            x,
            y,
            RelOp::Eq,
            true,
            true,
            |_fn_name, _arg, _other, _op, _is_lhs| {
                Some(UnaryInverseIsolationStepPlan {
                    equation: Equation {
                        lhs: x,
                        rhs: y,
                        op: RelOp::Eq,
                    },
                    items: vec![UnaryInverseExecutionItem {
                        equation: Equation {
                            lhs: x,
                            rhs: y,
                            op: RelOp::Eq,
                        },
                        description: "rewrite".to_string(),
                    }],
                    needs_rhs_cleanup: true,
                })
            },
            |_rhs| (z, vec![("cleanup".to_string(), z)]),
            |_lhs, rhs, _op| {
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![rhs]),
                    vec!["substep".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("inverse should be supported")
        .expect("pipeline should solve");

        assert!(matches!(solved.0, SolutionSet::Discrete(_)));
        assert_eq!(
            solved.1,
            vec![
                "rewrite".to_string(),
                "cleanup".to_string(),
                "substep".to_string()
            ]
        );
    }

    #[test]
    fn execute_unary_inverse_pipeline_with_items_with_returns_none_for_unsupported_inverse() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");

        let mut solve_calls = 0usize;
        let out = execute_unary_inverse_pipeline_with_items_with(
            "unknown",
            x,
            y,
            RelOp::Eq,
            true,
            true,
            |_fn_name, _arg, _other, _op, _is_lhs| None,
            |_rhs| (y, vec![]),
            |_lhs, _rhs, _op| {
                solve_calls += 1;
                Ok::<_, ()>((SolutionSet::AllReals, vec!["unexpected".to_string()]))
            },
            |item| item.description,
        );

        assert!(out.is_none());
        assert_eq!(solve_calls, 0);
    }

    #[test]
    fn collect_unary_inverse_solve_execution_items_preserves_order() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let execution = UnaryInverseSolveExecution {
            rewrite_items: vec![UnaryInverseExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: y,
                    op: RelOp::Eq,
                },
                description: "rewrite".to_string(),
            }],
            rhs_cleanup_items: vec![RhsSimplificationExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: z,
                    op: RelOp::Eq,
                },
                description: "cleanup".to_string(),
            }],
            rewritten_equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            target_rhs: z,
        };

        let items = collect_unary_inverse_solve_execution_items(&execution);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].description(), "rewrite");
        assert_eq!(items[1].description(), "cleanup");
        assert_eq!(items[0].equation.rhs, y);
        assert_eq!(items[1].equation.rhs, z);
    }
}
