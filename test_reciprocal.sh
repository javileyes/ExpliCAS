#!/bin/bash
cat << 'RUST' > /tmp/test_reciprocal.rs
use cas_ast::{Context, Expr};
use cas_parser::parse;

fn main() {
    let mut ctx = Context::new();
    
    // Parse the expressions
    let expr1 = parse("arctan(x) + arctan(1/x)", &mut ctx).unwrap();
    println!("Parsed: {:?}", ctx.get(expr1));
    
    // Let's drill down
    if let Expr::Add(l, r) = ctx.get(expr1) {
        println!("Left: {:?}", ctx.get(*l));
        println!("Right: {:?}", ctx.get(*r));
        
        if let Expr::Function(_, l_args) = ctx.get(*l) {
            println!(" Left arg: {:?}", ctx.get(l_args[0]));
        }
        if let Expr::Function(_, r_args) = ctx.get(*r) {
            println!("  Right arg: {:?}", ctx.get(r_args[0]));
            if let Expr::Div(num, den) = ctx.get(r_args[0]) {
                println!("    Numerator: {:?}", ctx.get(*num));
                println!("    Denominator: {:?}", ctx.get(*den));
            }
        }
    }
}
RUST
rustc --edition 2021 -L /Users/javiergimenezmoya/developer/math/target/debug/deps /tmp/test_reciprocal.rs -o /tmp/test_reciprocal --extern cas_ast=/Users/javiergimenezmoya/developer/math/target/debug/deps/libcas_ast.rlib --extern cas_parser=/Users/javiergimenezmoya/developer/math/target/debug/deps/libcas_parser.rlib --extern num_rational=/Users/javiergimenezmoya/developer/math/target/debug/deps/libnum_rational.rlib --extern num_bigint=/Users/javiergimenezmoya/developer/math/target/debug/deps/libnum_bigint.rlib 2>&1 | head -20
/tmp/test_reciprocal 2>&1 ||true
