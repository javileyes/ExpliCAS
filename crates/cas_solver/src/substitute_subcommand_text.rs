pub fn parse_substitute_wire_text_lines(
    payload: &str,
    steps_enabled: bool,
) -> Result<Vec<String>, String> {
    let value: serde_json::Value = serde_json::from_str(payload)
        .map_err(|e| format!("Invalid substitute wire payload: {e}"))?;

    let ok = value.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
    if !ok {
        let error_message = value
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .or_else(|| value.get("error").and_then(|e| e.as_str()))
            .unwrap_or("Substitute evaluation failed");
        return Err(error_message.to_string());
    }

    let result = value
        .get("result")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Missing result in substitute wire payload".to_string())?;

    let mut lines = Vec::new();
    if steps_enabled {
        if let Some(steps) = value.get("steps").and_then(|s| s.as_array()) {
            if !steps.is_empty() {
                lines.push("Steps:".to_string());
                for step in steps {
                    let before = step.get("before").and_then(|v| v.as_str()).unwrap_or("");
                    let after = step.get("after").and_then(|v| v.as_str()).unwrap_or("");
                    let rule = step
                        .get("rule")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Substitute");
                    let note = step.get("note").and_then(|v| v.as_str());
                    let line = match note {
                        Some(note) => format!("  {} → {} [{}] ({})", before, after, rule, note),
                        None => format!("  {} → {} [{}]", before, after, rule),
                    };
                    lines.push(line);
                }
            }
        }
    }
    lines.push(result.to_string());
    Ok(lines)
}
