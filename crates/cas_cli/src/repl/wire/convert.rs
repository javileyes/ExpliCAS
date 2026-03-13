use super::{WireKind, WireMsg, WireReply};
use crate::repl::output::{ReplMsg, ReplReply};
use std::borrow::Cow;
use std::path::Path;

fn file_action_text(prefix: &str, path: &Path) -> String {
    let display: Cow<'_, str> = path.as_os_str().to_string_lossy();
    let mut text = String::with_capacity(prefix.len() + display.len());
    text.push_str(prefix);
    text.push_str(&display);
    text
}

impl From<&ReplMsg> for WireMsg {
    fn from(msg: &ReplMsg) -> Self {
        match msg {
            ReplMsg::Output(text) => WireMsg {
                kind: WireKind::Output,
                text: text.clone(),
                span: None,
                data: None,
            },
            ReplMsg::Info(text) => WireMsg {
                kind: WireKind::Info,
                text: text.clone(),
                span: None,
                data: None,
            },
            ReplMsg::Warn(text) => WireMsg {
                kind: WireKind::Warn,
                text: text.clone(),
                span: None,
                data: None,
            },
            ReplMsg::Error(text) => WireMsg {
                kind: WireKind::Error,
                text: text.clone(),
                span: None,
                data: None,
            },
            ReplMsg::Steps(text) => WireMsg {
                kind: WireKind::Steps,
                text: text.clone(),
                span: None,
                data: None,
            },
            ReplMsg::Debug(text) => WireMsg {
                kind: WireKind::Debug,
                text: text.clone(),
                span: None,
                data: None,
            },
            ReplMsg::WriteFile { path, .. } => WireMsg {
                kind: WireKind::Info,
                text: file_action_text("Wrote: ", path),
                span: None,
                data: None,
            },
            ReplMsg::OpenFile { path } => WireMsg {
                kind: WireKind::Info,
                text: file_action_text("Open: ", path),
                span: None,
                data: None,
            },
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
    let mut messages = Vec::with_capacity(reply.len());
    for msg in reply {
        messages.push(WireMsg::from(msg));
    }
    WireReply::new(messages)
}
