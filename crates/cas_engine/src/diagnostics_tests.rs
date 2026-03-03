use crate::diagnostics::{Diagnostics, RequireOrigin};
use crate::implicit_domain::ImplicitCondition;
use cas_ast::Context;

#[test]
fn test_merge_origins() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let mut diag = Diagnostics::new();
    diag.push_required(
        ImplicitCondition::Positive(x),
        RequireOrigin::EquationImplicit,
    );
    diag.push_required(
        ImplicitCondition::Positive(x),
        RequireOrigin::EquationDerived,
    );

    assert_eq!(diag.requires.len(), 1, "Should merge same condition");
    assert_eq!(
        diag.requires[0].origins.len(),
        2,
        "Should have both origins"
    );
}

#[test]
fn test_trivial_filtered() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let x = ctx.var("x");

    let mut diag = Diagnostics::new();
    diag.push_required(
        ImplicitCondition::Positive(two),
        RequireOrigin::EquationImplicit,
    );
    diag.push_required(
        ImplicitCondition::Positive(x),
        RequireOrigin::OutputImplicit,
    );

    diag.dedup_and_sort(&ctx);

    assert_eq!(diag.requires.len(), 1, "Trivial 2 > 0 should be filtered");
    assert!(
        matches!(&diag.requires[0].cond, ImplicitCondition::Positive(e) if *e == x),
        "Only x > 0 should remain"
    );
}
