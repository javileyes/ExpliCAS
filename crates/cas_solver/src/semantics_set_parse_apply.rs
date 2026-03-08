use crate::semantics_set_parse_axis::set_semantic_axis;
use crate::semantics_set_types::SemanticsSetState;

/// Parse and apply arguments from `semantics set ...`.
pub fn evaluate_semantics_set_args(
    args: &[&str],
    mut state: SemanticsSetState,
) -> Result<SemanticsSetState, String> {
    if args.is_empty() {
        return Err("Usage: semantics set <axis> <value>\n\
                   or:  semantics set <axis>=<value> ..."
            .to_string());
    }

    let mut i = 0;
    while i < args.len() {
        let arg = args[i];

        if let Some((key, value)) = arg.split_once('=') {
            if let Some(err) = set_semantic_axis(&mut state, key, value) {
                return Err(err);
            }
            i += 1;
            continue;
        }

        if i + 1 >= args.len() {
            return Err(format!("ERROR: Missing value for axis '{}'", arg));
        }

        if arg == "solve" && args.get(i + 1) == Some(&"check") && i + 2 < args.len() {
            match args[i + 2] {
                "on" => state.check_solutions = true,
                "off" => state.check_solutions = false,
                other => {
                    return Err(format!(
                        "ERROR: Invalid value '{}' for 'solve check'\nAllowed: on, off",
                        other
                    ));
                }
            }
            i += 3;
            continue;
        }

        if let Some(err) = set_semantic_axis(&mut state, arg, args[i + 1]) {
            return Err(err);
        }
        i += 2;
    }

    Ok(state)
}
