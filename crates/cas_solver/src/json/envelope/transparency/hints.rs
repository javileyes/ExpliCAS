use crate::BlockedHint;
use cas_api_models::BlockedHintDto;

pub(super) fn map_blocked_hints(blocked_hints: &[BlockedHint]) -> Vec<BlockedHintDto> {
    blocked_hints
        .iter()
        .map(|h| BlockedHintDto {
            rule: h.rule.clone(),
            requires: vec![h.key.condition_display().to_string()],
            tip: h.suggestion.to_string(),
        })
        .collect()
}
