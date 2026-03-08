mod expansion;
mod mixed;
mod rationalization;
mod roots;
mod shared;

use crate::health_suite_types::HealthCase;

pub(crate) fn stress_suite() -> Vec<HealthCase> {
    let mut suite = Vec::new();
    suite.extend(expansion::stress_expansion_cases());
    suite.extend(rationalization::stress_rationalization_cases());
    suite.extend(roots::stress_root_cases());
    suite.extend(mixed::stress_mixed_cases());
    suite
}
