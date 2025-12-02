
#[test]
fn repro_expand_binomial_power_10() {
    use cas_engine::Simplifier;
    use cas_parser::parse;
    use std::time::Instant;

    let mut simplifier = Simplifier::new();
    simplifier.register_default_rules();
    simplifier.collect_steps = false; // Disable for benchmarking

    let expr = parse("(x + 1)^10", &mut simplifier.context).unwrap();
    
    let start = Instant::now();
    for _ in 0..100 {
        simplifier.simplify(expr);
    }
    let duration = start.elapsed();
    println!("expand_binomial_power_10 (100 runs): {:?}", duration);
    println!("Average: {:?}", duration / 100);
}

#[test]
fn repro_sum_fractions_10() {
    use cas_engine::Simplifier;
    use cas_parser::parse;
    use std::time::Instant;

    let mut simplifier = Simplifier::new();
    simplifier.register_default_rules();
    simplifier.collect_steps = false;

    // 1/x + 1/(x+1) + ... + 1/(x+9)
    let mut s = "1/x".to_string();
    for i in 1..10 {
        s.push_str(&format!(" + 1/(x+{})", i));
    }
    let expr = parse(&s, &mut simplifier.context).unwrap();
    
    let start = Instant::now();
    for _ in 0..10 { // Fewer runs as it's slower
        simplifier.simplify(expr);
    }
    let duration = start.elapsed();
    println!("sum_fractions_10 (10 runs): {:?}", duration);
    println!("Average: {:?}", duration / 10);
}
