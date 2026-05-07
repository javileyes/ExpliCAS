use crate::BlockedHint;
use cas_api_models::BlockedHintDto;

pub(super) fn map_blocked_hints(
    ctx: &cas_ast::Context,
    blocked_hints: &[BlockedHint],
) -> Vec<BlockedHintDto> {
    blocked_hints
        .iter()
        .map(|h| BlockedHintDto {
            rule: h.rule.clone(),
            requires: vec![crate::format_blocked_hint_condition(ctx, h)],
            tip: h.suggestion.to_string(),
        })
        .collect()
}
