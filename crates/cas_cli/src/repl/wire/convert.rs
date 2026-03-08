use super::{WireKind, WireMsg, WireReply};
use crate::repl::output::{ReplMsg, ReplReply};

impl From<&ReplMsg> for WireMsg {
    fn from(msg: &ReplMsg) -> Self {
        match msg {
            ReplMsg::Output(text) => WireMsg::new(WireKind::Output, text),
            ReplMsg::Info(text) => WireMsg::new(WireKind::Info, text),
            ReplMsg::Warn(text) => WireMsg::new(WireKind::Warn, text),
            ReplMsg::Error(text) => WireMsg::new(WireKind::Error, text),
            ReplMsg::Steps(text) => WireMsg::new(WireKind::Steps, text),
            ReplMsg::Debug(text) => WireMsg::new(WireKind::Debug, text),
            ReplMsg::WriteFile { path, .. } => {
                WireMsg::new(WireKind::Info, format!("Wrote: {}", path.display()))
            }
            ReplMsg::OpenFile { path } => {
                WireMsg::new(WireKind::Info, format!("Open: {}", path.display()))
            }
        }
    }
}

impl From<ReplMsg> for WireMsg {
    fn from(msg: ReplMsg) -> Self {
        WireMsg::from(&msg)
    }
}

/// Convert a REPL reply into the wire envelope.
pub fn wire_reply_from_repl(reply: &ReplReply) -> WireReply {
    let messages: Vec<WireMsg> = reply.iter().map(WireMsg::from).collect();
    WireReply::new(messages)
}
