/// Generate a short error ID for panic correlation.
///
/// Uses a hash of timestamp + message and returns 6 hex chars.
pub fn generate_short_error_id(msg: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut hasher = DefaultHasher::new();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    timestamp.hash(&mut hasher);
    msg.hash(&mut hasher);
    let hash = hasher.finish();

    format!("{:06X}", (hash & 0xFFFFFF) as u32)
}

/// Extract a stable panic message from a panic payload.
pub fn panic_payload_to_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Format a panic report line for debug logging.
pub fn format_panic_report_message(
    error_id: &str,
    version: &str,
    command: &str,
    panic_msg: &str,
) -> String {
    format!(
        "[PANIC_REPORT] id={} version={} command={:?} panic={}",
        error_id, version, command, panic_msg
    )
}

/// Format the user-facing panic message shown in the REPL.
pub fn format_user_panic_message(error_id: &str, panic_msg: &str) -> String {
    format!(
        "Internal error (id: {}): {}\n\n\
         The session is still active. You can continue working.\n\
         Please report this issue with the error id if it persists.",
        error_id, panic_msg
    )
}

#[cfg(test)]
mod tests {
    use super::{
        format_panic_report_message, format_user_panic_message, generate_short_error_id,
        panic_payload_to_message,
    };

    #[test]
    fn generate_short_error_id_returns_six_hex_chars() {
        let id = generate_short_error_id("panic message");
        assert_eq!(id.len(), 6);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn panic_payload_to_message_reads_string_payload() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(String::from("boom"));
        assert_eq!(
            panic_payload_to_message(payload.as_ref()),
            "boom".to_string()
        );
    }

    #[test]
    fn panic_payload_to_message_reads_str_payload() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("boom");
        assert_eq!(
            panic_payload_to_message(payload.as_ref()),
            "boom".to_string()
        );
    }

    #[test]
    fn panic_report_and_user_messages_include_error_id() {
        let report = format_panic_report_message("ABC123", "1.2.3", "solve x=1", "boom");
        assert!(report.contains("id=ABC123"));
        assert!(report.contains("version=1.2.3"));

        let user = format_user_panic_message("ABC123", "boom");
        assert!(user.contains("Internal error (id: ABC123): boom"));
        assert!(user.contains("session is still active"));
    }
}
