use crate::parent_context::ParentContext;
use cas_ast::{Context, Expr};
use num_traits::ToPrimitive;

#[test]
fn test_root_context() {
    let parent_ctx = ParentContext::root();

    assert_eq!(parent_ctx.immediate_parent(), None);
    assert_eq!(parent_ctx.all_ancestors().len(), 0);
    assert_eq!(parent_ctx.depth(), 0);
}

#[test]
fn test_with_parent() {
    let mut ctx = Context::new();
    let parent_id = ctx.num(42);
    let parent_ctx = ParentContext::with_parent(parent_id);

    assert_eq!(parent_ctx.immediate_parent(), Some(parent_id));
    assert_eq!(parent_ctx.all_ancestors().len(), 1);
    assert_eq!(parent_ctx.depth(), 1);
}

#[test]
fn test_extend() {
    let mut ctx = Context::new();
    let grandparent = ctx.num(1);
    let parent = ctx.num(2);

    let ctx1 = ParentContext::with_parent(grandparent);
    let ctx2 = ctx1.extend(parent);

    assert_eq!(ctx2.immediate_parent(), Some(parent));
    assert_eq!(ctx2.depth(), 2);
    assert_eq!(ctx2.all_ancestors(), &[grandparent, parent]);
}

#[test]
fn test_has_ancestor_matching() {
    let mut ctx = Context::new();
    let target = ctx.num(42);
    let _other = ctx.num(99);

    let parent_ctx = ParentContext::with_parent(target);

    assert!(parent_ctx.has_ancestor_matching(&ctx, |c, id| {
        matches!(c.get(id), Expr::Number(n) if n.to_i32() == Some(42))
    }));

    assert!(!parent_ctx.has_ancestor_matching(&ctx, |c, id| {
        matches!(c.get(id), Expr::Number(n) if n.to_i32() == Some(99))
    }));
}
