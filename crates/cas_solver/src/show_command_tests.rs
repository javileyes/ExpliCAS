#[cfg(test)]
mod tests {
    use crate::{
        evaluate_show_command_lines, evaluate_show_command_lines_with, Engine,
        InspectHistoryEntryInputError, ShowCommandContext,
    };

    #[test]
    fn evaluate_show_command_lines_with_invalid_id_reports_error() {
        let mut engine = Engine::new();
        let err = evaluate_show_command_lines_with(&mut engine, "show nope", |_engine, _line| {
            Err(InspectHistoryEntryInputError::InvalidId)
        })
        .expect_err("expected invalid id error");
        assert!(err.contains("Invalid entry ID"));
    }

    struct InvalidShowContext;

    impl ShowCommandContext for InvalidShowContext {
        fn inspect_history_entry_input(
            &mut self,
            _engine: &mut Engine,
            _input: &str,
        ) -> Result<crate::HistoryEntryInspection, InspectHistoryEntryInputError> {
            Err(InspectHistoryEntryInputError::InvalidId)
        }
    }

    #[test]
    fn evaluate_show_command_lines_invalid_id_reports_error() {
        let mut engine = Engine::new();
        let mut context = InvalidShowContext;
        let err = evaluate_show_command_lines(&mut context, &mut engine, "show nope")
            .expect_err("expected invalid id error");
        assert!(err.contains("Invalid entry ID"));
    }
}
