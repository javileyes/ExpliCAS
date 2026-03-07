use crate::health_suite_types::HealthCase;

use super::shared::stress_case;

pub(super) fn stress_expansion_cases() -> [HealthCase; 4] {
    [
        stress_case(
            "expand_product_chain",
            "expand((x+1)*(x+2)*(x+3))",
            200,
            350,
            120,
        ),
        stress_case("binomial_large", "(x+1)^8", 220, 450, 120),
        stress_case("distribute_sum", "3*(x+y+z+w)", 160, 250, 100),
        stress_case("nested_distribution", "2*(x + 3*(y+4))", 200, 300, 120),
    ]
}
