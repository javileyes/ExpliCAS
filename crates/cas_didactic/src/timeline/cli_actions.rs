use super::{TimelineCliAction, TimelineCliRender};

/// Convert timeline render output into primitive CLI actions.
pub fn timeline_cli_actions_from_render(render: TimelineCliRender) -> Vec<TimelineCliAction> {
    match render {
        TimelineCliRender::NoSteps { lines } => lines
            .into_iter()
            .map(TimelineCliAction::Output)
            .collect::<Vec<_>>(),
        TimelineCliRender::Html {
            file_name,
            html,
            lines,
        } => {
            let mut actions = vec![
                TimelineCliAction::WriteFile {
                    path: file_name.to_string(),
                    contents: html,
                },
                TimelineCliAction::OpenFile {
                    path: file_name.to_string(),
                },
            ];
            for line in lines {
                actions.push(TimelineCliAction::Output(line));
            }
            actions
        }
    }
}
