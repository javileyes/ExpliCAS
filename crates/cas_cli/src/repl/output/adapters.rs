use cas_didactic::TimelineCliAction;
use std::path::PathBuf;

use super::{ReplMsg, ReplReply};

/// Convert timeline CLI actions into a REPL reply payload.
pub fn timeline_cli_actions_to_reply(actions: Vec<TimelineCliAction>) -> ReplReply {
    let mut reply = ReplReply::new();
    for action in actions {
        match action {
            TimelineCliAction::Output(line) => reply.push(ReplMsg::output(line)),
            TimelineCliAction::WriteFile { path, contents } => {
                reply.push(ReplMsg::WriteFile {
                    path: PathBuf::from(path),
                    contents,
                });
            }
            TimelineCliAction::OpenFile { path } => {
                reply.push(ReplMsg::OpenFile {
                    path: PathBuf::from(path),
                });
            }
        }
    }
    reply
}

/// Convert a visualize command output into REPL actions.
pub fn visualize_output_to_reply(
    output: cas_solver::session_api::symbolic_commands::VisualizeCommandOutput,
) -> ReplReply {
    let mut reply = vec![ReplMsg::WriteFile {
        path: PathBuf::from(output.file_name),
        contents: output.dot_source,
    }];
    reply.extend(output.hint_lines.into_iter().map(ReplMsg::output));
    reply
}
