//! Test that Context::stats().nodes_created is correctly incremented
//!
//! Bug fixed: Context::add() was doing nodes.push() but NOT incrementing
//! stats.nodes_created, while add_raw() did. This caused budget tracking
//! to undercount nodes created via add().

use cas_ast::{Context, Expr};

#[test]
fn nodes_created_increments_on_new_var() {
    let mut ctx = Context::new();
    let before = ctx.stats().nodes_created;

    let _x = ctx.var("x");

    let after = ctx.stats().nodes_created;
    assert_eq!(
        after,
        before + 1,
        "Creating new variable should increment nodes_created"
    );
}

#[test]
fn nodes_created_does_not_increment_on_dedup() {
    let mut ctx = Context::new();
    let x1 = ctx.var("x");
    let after_first = ctx.stats().nodes_created;

    let x2 = ctx.var("x");
    let after_second = ctx.stats().nodes_created;

    assert_eq!(x1, x2, "Same variable should return same ExprId (dedup)");
    assert_eq!(
        after_first, after_second,
        "Deduplicated expression should NOT increment nodes_created"
    );
}

#[test]
fn nodes_created_increments_on_new_expression() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let after_var = ctx.stats().nodes_created;

    let _add = ctx.add(Expr::Add(x, x));
    let after_add = ctx.stats().nodes_created;

    assert!(
        after_add > after_var,
        "Creating new Add expression should increment nodes_created"
    );
}

#[test]
fn nodes_created_consistent_between_add_and_add_raw() {
    // Both add() and add_raw() should increment nodes_created when creating
    let mut ctx = Context::new();
    let before = ctx.stats().nodes_created;

    let x = ctx.var("x"); // Uses add()
    let after_add = ctx.stats().nodes_created;
    assert_eq!(after_add, before + 1, "add() should increment");

    let y = ctx.add_raw(Expr::Number(num_rational::BigRational::from_integer(
        42.into(),
    )));
    let after_add_raw = ctx.stats().nodes_created;
    assert_eq!(after_add_raw, after_add + 1, "add_raw() should increment");

    // Dedup on add_raw should not increment
    let _y2 = ctx.add_raw(Expr::Number(num_rational::BigRational::from_integer(
        42.into(),
    )));
    let after_dedup = ctx.stats().nodes_created;
    assert_eq!(after_dedup, after_add_raw, "Dedup should not increment");

    // Use x and y to avoid unused warnings
    let _ = (x, y);
}
