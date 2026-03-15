use crate::SemanticsSetState;

pub(super) fn set_reporting_axis(
    state: &mut SemanticsSetState,
    axis: &str,
    value: &str,
) -> Option<String> {
    match axis {
        "assumptions" => match value {
            "off" => state.assumption_reporting = crate::AssumptionReporting::Off,
            "summary" => state.assumption_reporting = crate::AssumptionReporting::Summary,
            "trace" => state.assumption_reporting = crate::AssumptionReporting::Trace,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'assumptions'\nAllowed: off, summary, trace",
                    value
                ));
            }
        },
        "assume_scope" => match value {
            "real" => state.assume_scope = crate::AssumeScope::Real,
            "wildcard" => state.assume_scope = crate::AssumeScope::Wildcard,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'assume_scope'\nAllowed: real, wildcard",
                    value
                ));
            }
        },
        "hints" => match value {
            "on" => state.hints_enabled = true,
            "off" => state.hints_enabled = false,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'hints'\nAllowed: on, off",
                    value
                ));
            }
        },
        "requires" => match value {
            "essential" => state.requires_display = crate::RequiresDisplayLevel::Essential,
            "all" => state.requires_display = crate::RequiresDisplayLevel::All,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'requires'\nAllowed: essential, all",
                    value
                ));
            }
        },
        _ => unreachable!("unsupported reporting axis: {axis}"),
    }

    None
}
