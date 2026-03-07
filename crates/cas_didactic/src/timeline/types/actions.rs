/// Normalized CLI actions derived from timeline render output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineCliAction {
    Output(String),
    WriteFile { path: String, contents: String },
    OpenFile { path: String },
}
