use crate::repl::{Repl, Verbosity};

use super::ReplReply;

/// Delta of UI state changes produced by a core operation.
/// The Repl wrapper applies these changes after processing the reply.
#[derive(Debug, Clone, Default)]
pub struct UiDelta {
    /// New verbosity level (if changed by command like `set steps`)
    pub verbosity: Option<Verbosity>,
}

/// Result from a core operation: reply messages + optional UI state changes.
#[derive(Debug, Clone)]
pub struct CoreResult {
    /// Messages to display
    pub reply: ReplReply,
    /// UI state changes to apply
    pub ui_delta: UiDelta,
}

impl CoreResult {
    /// Create a CoreResult with only reply, no UI changes
    pub fn reply_only(reply: ReplReply) -> Self {
        Self {
            reply,
            ui_delta: UiDelta::default(),
        }
    }

    /// Create a CoreResult with reply and UI delta
    pub fn with_delta(reply: ReplReply, ui_delta: UiDelta) -> Self {
        Self { reply, ui_delta }
    }
}

impl From<ReplReply> for CoreResult {
    fn from(reply: ReplReply) -> Self {
        Self::reply_only(reply)
    }
}

impl Repl {
    /// Apply UI delta mutations (e.g. verbosity) emitted by core handlers.
    pub(crate) fn apply_ui_delta(&mut self, ui_delta: UiDelta) {
        if let Some(verbosity) = ui_delta.verbosity {
            self.verbosity = verbosity;
        }
    }

    /// Apply UI delta and return the reply payload for printing.
    pub(crate) fn finalize_core_result(&mut self, result: CoreResult) -> ReplReply {
        self.apply_ui_delta(result.ui_delta);
        result.reply
    }
}
