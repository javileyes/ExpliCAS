#[cfg(test)]
mod tests {
    use crate::limit_subcommand::{
        evaluate_limit_subcommand, LimitCommandApproach, LimitCommandPreSimplify,
        LimitSubcommandOutput,
    };

    #[test]
    fn evaluate_limit_subcommand_json_contract() {
        let out = evaluate_limit_subcommand(
            "1/x",
            "x",
            LimitCommandApproach::Infinity,
            LimitCommandPreSimplify::Safe,
            true,
        )
        .expect("limit json");

        match out {
            LimitSubcommandOutput::Json(payload) => {
                assert!(payload.contains("\"ok\""));
            }
            _ => panic!("expected json output"),
        }
    }
}
