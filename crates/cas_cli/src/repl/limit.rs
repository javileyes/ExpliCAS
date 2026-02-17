use super::*;

impl Repl {
    pub(crate) fn handle_limit(&mut self, line: &str) {
        let reply = self.handle_limit_core(line);
        self.print_reply(reply);
    }

    fn handle_limit_core(&mut self, line: &str) -> ReplReply {
        use cas_solver::{limit, Approach, Budget, LimitOptions, PreSimplifyMode};

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

        // Parse: expr [, var [, direction [, mode]]]
        // Split by comma, respecting parentheses
        let parts: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();

        let expr_str = parts.first().unwrap_or(&"");
        let var_str = parts.get(1).copied().unwrap_or("x");
        let dir_str = parts.get(2).copied().unwrap_or("infinity");
        let mode_str = parts.get(3).copied().unwrap_or("off");

        // Parse expression
        let expr = match cas_parser::parse(expr_str, &mut self.core.engine.simplifier.context) {
            Ok(e) => e,
            Err(e) => {
                return reply_output(format!("Parse error: {:?}", e));
            }
        };

        // Get variable
        let var = self.core.engine.simplifier.context.var(var_str);

        // Parse direction
        let approach = if dir_str.contains("-infinity") || dir_str.contains("-inf") {
            Approach::NegInfinity
        } else {
            Approach::PosInfinity
        };

        // Parse presimplify mode
        let presimplify = if mode_str.eq_ignore_ascii_case("safe") {
            PreSimplifyMode::Safe
        } else {
            PreSimplifyMode::Off
        };

        // Compute limit
        let mut budget = Budget::new();
        let opts = LimitOptions {
            presimplify,
            ..Default::default()
        };

        match limit(
            &mut self.core.engine.simplifier.context,
            expr,
            var,
            approach,
            &opts,
            &mut budget,
        ) {
            Ok(result) => {
                let result_disp = cas_formatter::DisplayExpr {
                    context: &self.core.engine.simplifier.context,
                    id: result.expr,
                };

                let dir_disp = match approach {
                    Approach::PosInfinity => "+∞",
                    Approach::NegInfinity => "-∞",
                };

                let mut lines = vec![format!(
                    "lim_{{{}→{}}} = {}",
                    var_str, dir_disp, result_disp
                )];

                if let Some(warning) = result.warning {
                    lines.push(format!("Warning: {}", warning));
                }

                reply_output(lines.join("\n"))
            }
            Err(e) => reply_output(format!("Error computing limit: {}", e)),
        }
    }
}
