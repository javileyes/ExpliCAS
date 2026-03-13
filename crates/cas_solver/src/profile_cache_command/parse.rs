use super::ProfileCacheCommandInput;

pub(super) fn parse_profile_cache_command_input(line: &str) -> ProfileCacheCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1).copied() {
        None | Some("status") => ProfileCacheCommandInput::Status,
        Some("clear") => ProfileCacheCommandInput::Clear,
        Some(other) => ProfileCacheCommandInput::Unknown(other.to_string()),
    }
}
