use super::*;

impl Repl {
    pub(crate) fn handle_limit(&mut self, line: &str) {
        use cas_engine::limits::{limit, Approach, LimitOptions};
        use cas_engine::Budget;

        let rest = line.strip_prefix("limit").unwrap_or(line).trim();
        if rest.is_empty() {
            println!("Usage: limit <expr> [, <var> [, <direction> [, safe]]]");
            println!("Examples:");
            println!("  limit x^2                      → infinity (default: x → +∞)");
            println!("  limit (x^2+1)/(2*x^2-3), x     → 1/2");
            println!("  limit x^3/x^2, x, -infinity    → -infinity");
            println!("  limit (x-x)/x, x, infinity, safe → 0 (with pre-simplify)");
            return;
        }

        // Parse: expr [, var [, direction [, mode]]]
        // Split by comma, respecting parentheses
        let parts: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();

        let expr_str = parts.first().unwrap_or(&"");
        let var_str = parts.get(1).copied().unwrap_or("x");
        let dir_str = parts.get(2).copied().unwrap_or("infinity");
        let mode_str = parts.get(3).copied().unwrap_or("off");

        // Parse expression
        let expr = match cas_parser::parse(expr_str, &mut self.engine.simplifier.context) {
            Ok(e) => e,
            Err(e) => {
                println!("Parse error: {:?}", e);
                return;
            }
        };

        // Get variable
        let var = self.engine.simplifier.context.var(var_str);

        // Parse direction
        let approach = if dir_str.contains("-infinity") || dir_str.contains("-inf") {
            Approach::NegInfinity
        } else {
            Approach::PosInfinity
        };

        // Parse presimplify mode
        let presimplify = if mode_str.eq_ignore_ascii_case("safe") {
            cas_engine::limits::PreSimplifyMode::Safe
        } else {
            cas_engine::limits::PreSimplifyMode::Off
        };

        // Compute limit
        let mut budget = Budget::new();
        let opts = LimitOptions {
            presimplify,
            ..Default::default()
        };

        match limit(
            &mut self.engine.simplifier.context,
            expr,
            var,
            approach,
            &opts,
            &mut budget,
        ) {
            Ok(result) => {
                let result_disp = cas_ast::DisplayExpr {
                    context: &self.engine.simplifier.context,
                    id: result.expr,
                };

                let dir_disp = match approach {
                    Approach::PosInfinity => "+∞",
                    Approach::NegInfinity => "-∞",
                };

                println!("lim_{{{}→{}}} = {}", var_str, dir_disp, result_disp);

                if let Some(warning) = result.warning {
                    println!("Warning: {}", warning);
                }
            }
            Err(e) => {
                println!("Error computing limit: {}", e);
            }
        }
    }
}
