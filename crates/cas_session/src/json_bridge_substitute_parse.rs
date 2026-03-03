use crate::json_bridge_substitute_types::{ParseField, SubstituteParseIssue};
use cas_api_models::SpanJson as ApiSpanJson;
use cas_ast::{Context, ExprId};

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

pub(crate) fn parse_substitute_input(
    expr_str: &str,
    target_str: &str,
    replacement_str: &str,
) -> Result<(Context, ExprId, ExprId, ExprId), SubstituteParseIssue> {
    let mut ctx = Context::new();
    let expr = parse_component(expr_str, &mut ctx, ParseField::Expression)?;
    let target = parse_component(target_str, &mut ctx, ParseField::Target)?;
    let replacement = parse_component(replacement_str, &mut ctx, ParseField::Replacement)?;
    Ok((ctx, expr, target, replacement))
}
