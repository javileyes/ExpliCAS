mod baseline;
mod expansion;
mod fractions;
mod mixed;
mod powers;
mod rationalization;
mod roots;
mod transform;

use crate::health_suite_models::HealthCase;

use self::baseline::baseline_cases;
use self::expansion::expansion_cases;
use self::fractions::fractions_cases;
use self::mixed::mixed_cases;
use self::powers::powers_cases;
use self::rationalization::rationalization_cases;
use self::roots::roots_cases;
use self::transform::transform_cases;

pub(crate) fn core_suite() -> Vec<HealthCase> {
    let mut suite = Vec::new();
    suite.extend(transform_cases());
    suite.extend(expansion_cases());
    suite.extend(fractions_cases());
    suite.extend(rationalization_cases());
    suite.extend(mixed_cases());
    suite.extend(baseline_cases());
    suite.extend(roots_cases());
    suite.extend(powers_cases());
    suite
}
