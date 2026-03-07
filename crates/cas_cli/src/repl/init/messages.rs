use super::*;

impl Repl {
    /// Generate startup banner messages (no I/O here)
    pub(crate) fn startup_messages(&self) -> ReplReply {
        reply_output(
            "Rust CAS Step-by-Step Demo\n\
             Step-by-step output enabled (Normal).\n\
             Enter an expression (e.g., '2 * 3 + 0'):",
        )
    }

    /// Generate goodbye message (no I/O here)
    pub(crate) fn goodbye_message(&self) -> ReplReply {
        reply_output("Goodbye!")
    }
}
