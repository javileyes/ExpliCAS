use cas_api_models::{EngineJsonError as ApiEngineJsonError, SpanJson as ApiSpanJson};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParseField {
    Expression,
    Target,
    Replacement,
}

#[derive(Clone, Debug)]
pub(crate) struct SubstituteParseIssue {
    pub(crate) field: ParseField,
    pub(crate) error: String,
    pub(crate) span: Option<ApiSpanJson>,
}

impl SubstituteParseIssue {
    pub(crate) fn to_json_error(&self) -> ApiEngineJsonError {
        let message = match self.field {
            ParseField::Expression => format!("Failed to parse expression: {}", self.error),
            ParseField::Target => format!("Failed to parse target: {}", self.error),
            ParseField::Replacement => format!("Failed to parse replacement: {}", self.error),
        };
        ApiEngineJsonError::parse(message, self.span)
    }
}
