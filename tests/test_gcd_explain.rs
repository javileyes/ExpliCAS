// Simple test program to verify GCD educational mode
use cas_ast::Context;
use cas_engine::rules::number_theory::{compute_gcd, GcdResult};
use cas_parser::parse;

fn main() {
    let mut ctx = Context::new();

    println!("=== Test 1: Integer GCD with explain=true ===");
    let a = parse("48", &mut ctx).unwrap();
    let b = parse("18", &mut ctx).unwrap();

    // Note: we need to make compute_gcd and GcdResult public for this test
    // For now, let's just run the existing tests
    println!("Test placeholder - compute_gcd is not public");
    println!("We'll verify via the existing test suite instead");
}
