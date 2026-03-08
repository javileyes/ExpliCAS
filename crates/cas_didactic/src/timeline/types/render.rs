/// CLI-facing timeline render artifact.
#[derive(Debug, Clone)]
pub enum TimelineCliRender {
    /// No timeline file should be emitted; return textual lines only.
    NoSteps { lines: Vec<String> },
    /// Timeline file + informational lines.
    Html {
        file_name: &'static str,
        html: String,
        lines: Vec<String>,
    },
}
