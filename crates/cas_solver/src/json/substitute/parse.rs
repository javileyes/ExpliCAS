use cas_api_models::{EngineJsonError as ApiEngineJsonError, SpanJson as ApiSpanJson};
use cas_ast::{Context, ExprId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseField {
    Expression,
    Target,
    Replacement,
}

#[derive(Clone, Debug)]
pub struct SubstituteParseIssue {
    pub field: ParseField,
    pub error: String,
    pub span: Option<ApiSpanJson>,
}

impl SubstituteParseIssue {
    pub fn to_json_error(&self) -> ApiEngineJsonError {
        let message = match self.field {
            ParseField::Expression => format!("Failed to parse expression: {}", self.error),
            ParseField::Target => format!("Failed to parse target: {}", self.error),
            ParseField::Replacement => format!("Failed to parse replacement: {}", self.error),
        };
        ApiEngineJsonError::parse(message, self.span)
    }
}

fn parse_component(
    input: &str,
    ctx: &mut Context,
    field: ParseField,
) -> Result<ExprId, SubstituteParseIssue> {
    cas_parser::parse(input, ctx).map_err(|e| SubstituteParseIssue {
        field,
        error: e.to_string(),
        span: e.span().map(|s| ApiSpanJson {
            start: s.start,
            end: s.end,
        }),
    })
}

pub fn parse_substitute_input(
    expr_str: &str,
    target_str: &str,
    with_str: &str,
) -> Result<(Context, ExprId, ExprId, ExprId), SubstituteParseIssue> {
    let mut ctx = Context::new();
    let expr = parse_component(expr_str, &mut ctx, ParseField::Expression)?;
    let target = parse_component(target_str, &mut ctx, ParseField::Target)?;
    let replacement = parse_component(with_str, &mut ctx, ParseField::Replacement)?;
    Ok((ctx, expr, target, replacement))
}
