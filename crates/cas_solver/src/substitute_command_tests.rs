#[cfg(test)]
mod tests {
    use crate::substitute_command_eval::evaluate_substitute_invocation_user_message;
    use crate::substitute_command_format::{
        format_substitute_parse_error_message, substitute_render_mode_from_display_mode,
    };
    use cas_api_models::{SubstituteParseError, SubstituteRenderMode};

    #[test]
    fn substitute_render_mode_maps_from_set_display_mode() {
        assert_eq!(
            substitute_render_mode_from_display_mode(crate::SetDisplayMode::Verbose),
            SubstituteRenderMode::Verbose
        );
    }

    #[test]
    fn format_substitute_parse_error_message_usage_is_human_readable() {
        let msg = format_substitute_parse_error_message(&SubstituteParseError::InvalidArity);
        assert!(msg.contains("Usage: subst"));
    }

    #[test]
    fn evaluate_substitute_invocation_user_message_formats_parse_errors() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let message = evaluate_substitute_invocation_user_message(
            &mut simplifier,
            "subst x^2 + x",
            crate::SetDisplayMode::Normal,
        )
        .expect_err("invalid arity");
        assert!(message.contains("Usage: subst"));
    }
}
