use super::*;

impl Repl {
    pub(crate) fn handle_limit(&mut self, line: &str) {
        let reply = self.handle_limit_core(line);
        self.print_reply(reply);
    }

    fn handle_limit_core(&mut self, line: &str) -> ReplReply {
        use cas_solver::Approach;

        let rest = line.strip_prefix("limit").unwrap_or(line).trim();
        if rest.is_empty() {
            return reply_output(
                "Usage: limit <expr> [, <var> [, <direction> [, safe]]]\n\
                 Examples:\n\
                   limit x^2                      → infinity (default: x → +∞)\n\
                   limit (x^2+1)/(2*x^2-3), x     → 1/2\n\
                   limit x^3/x^2, x, -infinity    → -infinity\n\
                   limit (x-x)/x, x, infinity, safe → 0 (with pre-simplify)",
            );
        }

        let parsed = cas_solver::parse_limit_command_input(rest);
        let expr_str = parsed.expr;
        let var_str = parsed.var;
        let approach = parsed.approach;
        let presimplify = parsed.presimplify;

        match cas_solver::json::eval_limit_from_str(expr_str, var_str, approach, presimplify) {
            Ok(limit_result) => {
                let dir_disp = match approach {
                    Approach::PosInfinity => "+∞",
                    Approach::NegInfinity => "-∞",
                };
                let mut lines = vec![format!(
                    "lim_{{{}→{}}} = {}",
                    var_str, dir_disp, limit_result.result
                )];
                if let Some(warning) = limit_result.warning {
                    lines.push(format!("Warning: {}", warning));
                }
                reply_output(lines.join("\n"))
            }
            Err(cas_solver::json::LimitEvalError::Parse(message)) => reply_output(message),
            Err(cas_solver::json::LimitEvalError::Limit(message)) => {
                reply_output(format!("Error computing limit: {}", message))
            }
        }
    }
}
