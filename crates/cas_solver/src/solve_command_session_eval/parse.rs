#[derive(Debug, Clone)]
pub(super) struct SolveCommandSessionRenderRequest {
    pub check_enabled: bool,
    pub parsed: crate::SolveCommandInput,
}

pub(super) fn parse_solve_command_session_request(
    line: &str,
    eval_options: &crate::EvalOptions,
) -> SolveCommandSessionRenderRequest {
    let rest = line.strip_prefix("solve").unwrap_or(line).trim();
    let (check_enabled, solve_tail) =
        crate::parse_solve_invocation_check(rest, eval_options.check_solutions);

    SolveCommandSessionRenderRequest {
        check_enabled,
        parsed: crate::parse_solve_command_input(solve_tail),
    }
}
