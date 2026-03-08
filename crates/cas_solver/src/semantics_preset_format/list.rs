use crate::semantics_preset_catalog::semantics_presets;

pub fn format_semantics_preset_list_lines() -> Vec<String> {
    let mut lines = Vec::new();
    lines.push("Available presets:".to_string());
    for preset in semantics_presets() {
        lines.push(format!("  {:10} {}", preset.name, preset.description));
    }
    lines.push(String::new());
    lines.push("Usage:".to_string());
    lines.push("  semantics preset <name>       Apply preset".to_string());
    lines.push("  semantics preset help <name>  Show preset axes".to_string());
    lines
}
