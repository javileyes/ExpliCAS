use cas_api_models::LimitSubcommandEvalError;

pub fn format_limit_subcommand_error(error: &LimitSubcommandEvalError) -> String {
    match error {
        LimitSubcommandEvalError::Parse(message) => message.clone(),
        LimitSubcommandEvalError::Limit(message) => format!("Error: {message}"),
    }
}
