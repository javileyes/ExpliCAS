mod test_utils;

use test_utils::assert_simplifies_to;

#[test]
fn geometric_difference_product_collapses_without_expansion_bloat() {
    assert_simplifies_to("(x - 1)*(x^5 + x^4 + x^3 + x^2 + x + 1)", "x^6 - 1");
}
