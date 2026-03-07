use super::super::types::TimelineCliRender;
use super::{TIMELINE_HTML_FILE, TIMELINE_OPEN_HINT_MESSAGE};

pub(super) fn render_no_steps(lines: Vec<String>) -> TimelineCliRender {
    TimelineCliRender::NoSteps { lines }
}

pub(super) fn render_html(html: String, mut lines: Vec<String>) -> TimelineCliRender {
    if !lines.iter().any(|line| line == TIMELINE_OPEN_HINT_MESSAGE) {
        lines.push(TIMELINE_OPEN_HINT_MESSAGE.to_string());
    }

    TimelineCliRender::Html {
        file_name: TIMELINE_HTML_FILE,
        html,
        lines,
    }
}
