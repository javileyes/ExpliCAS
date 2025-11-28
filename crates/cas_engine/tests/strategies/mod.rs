use proptest::prelude::*;
use cas_ast::Expr;
use std::rc::Rc;

pub fn arb_expr() -> impl Strategy<Value = Rc<Expr>> {
    let leaf = prop_oneof![
        // Numbers: small integers for simplicity
        (-10i64..10).prop_map(|n| Expr::num(n)),
        // Variables
        "[a-z]".prop_map(|s| Expr::var(&s)),
        // Constants
        Just(Expr::pi()),
        Just(Expr::e()),
    ];

    leaf.prop_recursive(
        4, // levels deep
        64, // max size
        10, // items per collection
        |inner| prop_oneof![
            // Binary ops
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::add(l, r)),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::sub(l, r)),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::mul(l, r)),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::div(l, r)),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::pow(l, r)),
            // Unary ops
            inner.clone().prop_map(|e| Expr::neg(e)),
            // Functions (simple ones)
            inner.clone().prop_map(|e| Expr::sin(e)),
            inner.clone().prop_map(|e| Expr::cos(e)),
            inner.clone().prop_map(|e| Expr::ln(e)),
        ]
    )
}
