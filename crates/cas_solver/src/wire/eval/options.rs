use cas_api_models::{BudgetWireInfo, EngineWireResponse, EvalRunOptions};

pub(super) fn parse_eval_run_options(opts_json: &str) -> Result<EvalRunOptions, String> {
    let trimmed = opts_json.trim();
    if let Some(opts) = parse_eval_run_options_fast(trimmed) {
        return Ok(opts);
    }

    match serde_json::from_str(trimmed) {
        Ok(opts) => Ok(opts),
        Err(e) => {
            let resp = EngineWireResponse::invalid_options_json(e.to_string());
            Err(resp.to_json_with_pretty(EvalRunOptions::requested_pretty(trimmed)))
        }
    }
}

fn parse_eval_run_options_fast(opts_json: &str) -> Option<EvalRunOptions> {
    match opts_json {
        "{}"
        | r#"{"pretty":false}"#
        | r#"{"steps":false}"#
        | r#"{"steps":false,"pretty":false}"#
        | r#"{"pretty":false,"steps":false}"#
        | r#"{"budget":{"preset":"cli"}}"#
        | r#"{"budget":{"preset":"cli","mode":"best-effort"}}"#
        | r#"{"budget":{"mode":"best-effort","preset":"cli"}}"# => Some(EvalRunOptions::default()),
        r#"{"budget":{"preset":"cli","mode":"strict"}}"#
        | r#"{"budget":{"mode":"strict","preset":"cli"}}"# => Some(EvalRunOptions {
            budget: cas_api_models::BudgetOpts {
                preset: "cli".to_string(),
                mode: "strict".to_string(),
            },
            ..EvalRunOptions::default()
        }),
        r#"{"pretty":true}"# => Some(EvalRunOptions {
            pretty: true,
            ..EvalRunOptions::default()
        }),
        r#"{"steps":true}"# => Some(EvalRunOptions {
            steps: true,
            ..EvalRunOptions::default()
        }),
        r#"{"steps":true,"pretty":true}"# | r#"{"pretty":true,"steps":true}"# => {
            Some(EvalRunOptions {
                steps: true,
                pretty: true,
                ..EvalRunOptions::default()
            })
        }
        r#"{"steps":true,"pretty":false}"# | r#"{"pretty":false,"steps":true}"# => {
            Some(EvalRunOptions {
                steps: true,
                ..EvalRunOptions::default()
            })
        }
        r#"{"steps":false,"pretty":true}"# | r#"{"pretty":true,"steps":false}"# => {
            Some(EvalRunOptions {
                pretty: true,
                ..EvalRunOptions::default()
            })
        }
        r#"{"budget":{"preset":"cli"},"pretty":true}"#
        | r#"{"pretty":true,"budget":{"preset":"cli"}}"#
        | r#"{"budget":{"preset":"cli","mode":"best-effort"},"pretty":true}"#
        | r#"{"pretty":true,"budget":{"preset":"cli","mode":"best-effort"}}"#
        | r#"{"budget":{"mode":"best-effort","preset":"cli"},"pretty":true}"#
        | r#"{"pretty":true,"budget":{"mode":"best-effort","preset":"cli"}}"# => {
            Some(EvalRunOptions {
                pretty: true,
                ..EvalRunOptions::default()
            })
        }
        _ => None,
    }
}

pub(super) fn build_budget_info(opts: &EvalRunOptions) -> BudgetWireInfo {
    let strict = opts.budget.mode == "strict";
    BudgetWireInfo::new(&opts.budget.preset, strict)
}

#[cfg(test)]
mod tests {
    use super::parse_eval_run_options;

    #[test]
    fn parse_eval_run_options_fast_default_and_flags() {
        let opts = parse_eval_run_options("{}").expect("parse failed");
        assert!(!opts.pretty);
        assert!(!opts.steps);
        assert_eq!(opts.budget.preset, "cli");
        assert_eq!(opts.budget.mode, "best-effort");

        let opts = parse_eval_run_options(r#"{"pretty":true}"#).expect("parse failed");
        assert!(opts.pretty);
        assert!(!opts.steps);

        let opts = parse_eval_run_options(r#"{"steps":true,"pretty":true}"#).expect("parse failed");
        assert!(opts.pretty);
        assert!(opts.steps);
    }

    #[test]
    fn parse_eval_run_options_fast_budget_cli_default_shapes() {
        let opts = parse_eval_run_options(r#"{"budget":{"preset":"cli"}}"#).expect("parse failed");
        assert_eq!(opts.budget.preset, "cli");
        assert_eq!(opts.budget.mode, "best-effort");

        let opts = parse_eval_run_options(r#"{"budget":{"mode":"best-effort","preset":"cli"}}"#)
            .expect("parse failed");
        assert_eq!(opts.budget.preset, "cli");
        assert_eq!(opts.budget.mode, "best-effort");
    }

    #[test]
    fn parse_eval_run_options_falls_back_for_non_fast_shapes() {
        let opts =
            parse_eval_run_options(r#"{"budget":{"preset":"cli","mode":"strict"},"steps":true}"#)
                .expect("parse failed");
        assert_eq!(opts.budget.mode, "strict");
        assert!(opts.steps);
    }

    #[test]
    fn parse_eval_run_options_fast_budget_strict_shapes() {
        let opts = parse_eval_run_options(r#"{"budget":{"preset":"cli","mode":"strict"}}"#)
            .expect("parse failed");
        assert_eq!(opts.budget.preset, "cli");
        assert_eq!(opts.budget.mode, "strict");

        let opts = parse_eval_run_options(r#"{"budget":{"mode":"strict","preset":"cli"}}"#)
            .expect("parse failed");
        assert_eq!(opts.budget.preset, "cli");
        assert_eq!(opts.budget.mode, "strict");
    }
}
