
#[test]
fn test_canonicalized_conj

ugate_add_add() {
    let mut ctx = Context::new();
    // After canonicalization, (x-1)*(x+1) becomes (-1+x)*(1+x)
    // This should still be detected as canonical
    let x = ctx.var("x");
    let one = ctx.num(1);
    let neg_one = ctx.num(-1);
    
    // (-1 + x) * (1 + x)
    let left = ctx.add(Expr::Add(neg_one, x));  // -1 + x
    let right = ctx.add(Expr::Add(one, x));      // 1 + x
    let product = ctx.add(Expr::Mul(left, right));
    
    assert!(
        is_canonical_form(&ctx, product),
        "(-1+x)*(1+x) should be canonical (they are conjugates)"
    );
}
