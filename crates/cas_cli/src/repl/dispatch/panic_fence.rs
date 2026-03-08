use super::*;

impl Repl {
    /// Main command dispatch - calls core and prints result.
    ///
    /// Wraps command execution with `catch_unwind` to prevent internal panics
    /// from crashing the REPL session.
    pub fn handle_command(&mut self, line: &str) {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        let result = catch_unwind(AssertUnwindSafe(|| self.handle_command_core(line)));

        match result {
            Ok(reply) => self.print_reply(reply),
            Err(panic_info) => {
                let panic_msg =
                    super::super::panic_guard::panic_payload_to_message(panic_info.as_ref());
                let error_id = super::super::panic_guard::generate_short_error_id(&panic_msg);

                if std::env::var("EXPLICAS_PANIC_REPORT").is_ok() {
                    let version = env!("CARGO_PKG_VERSION");
                    let log_msg = super::super::panic_guard::format_panic_report_message(
                        &error_id, version, line, &panic_msg,
                    );
                    self.print_reply(vec![ReplMsg::Debug(log_msg)]);
                }

                self.print_reply(vec![ReplMsg::Error(
                    super::super::panic_guard::format_user_panic_message(&error_id, &panic_msg),
                )]);
            }
        }
    }
}
