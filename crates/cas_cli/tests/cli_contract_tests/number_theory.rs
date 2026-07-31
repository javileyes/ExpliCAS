use super::*;

#[test]
fn test_eval_unimodular_abs() {
    // Fase 2 · residual B2 cerrado con disciplina V0: `|cos θ ± i·sin θ| = 1` SOLO con
    // θ constante real DECIDIBLE (provable_const_sign). Una variable DEBE declinar:
    // bajo ComplexEnabled puede tomar valor complejo y la unimodularidad es falsa
    // (x:=i ⇒ |e^(i·i)| = 1/e ≠ 1) — el mismo sticky-fold que V0 mató en norm.
    let rc = |input: &str| -> String {
        let out = cli()
            .args([
                "eval",
                input,
                "--value-domain",
                "complex",
                "--format",
                "json",
            ])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Graduates: literal, surd (const_sign prueba sqrt(2)), conjugado, vía Euler.
    assert_eq!(rc("abs(e^(2*i))"), "1");
    assert_eq!(rc("abs(cos(2)+i*sin(2))"), "1");
    assert_eq!(rc("abs(e^(i*sqrt(2)))"), "1");
    assert_eq!(rc("abs(cos(2)-i*sin(2))"), "1");
    assert_eq!(rc("abs(e^(-2*i))"), "1");
    // Declines V0-discipline: símbolo (puede ser complejo), θ distinto, θ=i, real mode.
    assert_eq!(rc("abs(cos(x)+i*sin(x))"), "|cos(x) + i·sin(x)|");
    // El mismatch θ≠θ' dejó de ser residual en tanda-3 ciclo 2: el módulo
    // Gaussiano-surd lo computa por la vía general (const_sign acota trig de
    // racionales) → √(sin(3)²+cos(2)²), correcto y más informativo. La propiedad
    // de ESTE test (unimodularidad solo con θ IGUAL y decidible) sigue fijada
    // por los asserts de arriba.
    assert_eq!(rc("abs(cos(2)+i*sin(3))"), "sqrt(sin(3)^2 + cos(2)^2)");
    // θ=i: la unimodularidad sigue DECLINANDO aquí, pero el puente trig-de-i
    // (ciclo 3) compone |cosh(1) − sinh(1)| = 1/e — exactamente el contraejemplo
    // |e^(i·i)| = 1/e que motiva el guard de realidad. El valor confirma la
    // disciplina: NO es 1.
    assert_eq!(rc("abs(cos(i)+i*sin(i))"), "1 / e");
    assert_eq!(r("abs(cos(2)+i*sin(2))"), "|cos(2) + i·sin(2)|");
}
#[test]
fn test_eval_number_theory_divisors_and_crt() {
    // `divisors(n)` lists the positive divisors (sorted), and `crt` solves a system of congruences
    // (Chinese Remainder Theorem), declining on an inconsistent non-coprime system. sympy-checked.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("divisors(12)"), "[1, 2, 3, 4, 6, 12]");
    assert_eq!(r("divisors(7)"), "[1, 7]");
    assert_eq!(r("divisors(36)"), "[1, 2, 3, 4, 6, 9, 12, 18, 36]");
    assert_eq!(r("crt([2,3],[3,5])"), "8"); // x≡2 (mod 3), x≡3 (mod 5)
    assert_eq!(r("crt([1,2,3],[2,3,5])"), "23");
    // Inconsistent congruences with non-coprime moduli ⇒ honest residual.
    assert_eq!(r("crt([2,4],[3,6])"), "crt([[2], [4]], [[3], [6]])");
}
#[test]
fn test_eval_number_theory_gcdext() {
    // `gcdext(a,b)` (aliases `bezout`/`xgcd`) returns [g, x, y] with a·x + b·y = g = gcd(a,b) — the
    // Bézout coefficients from extended Euclid.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("gcdext(12,18)"), "[6, -1, 1]"); // 12·(-1) + 18·1 = 6
    assert_eq!(r("gcdext(3,7)"), "[1, -2, 1]"); // 3·(-2) + 7·1 = 1
    assert_eq!(r("gcdext(48,36)"), "[12, 1, -1]"); // 48·1 + 36·(-1) = 12
    assert_eq!(r("gcdext(17,5)"), "[1, -2, 7]");
}
#[test]
fn test_eval_number_theory_modular() {
    // Modular arithmetic: modinv (modular inverse via extended Euclid, residual when gcd≠1) and the
    // Jacobi symbol (−1/0/1). Cross-checked against sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("modinv(3,7)"), "5"); // 3·5 = 15 ≡ 1 (mod 7)
    assert_eq!(r("modinv(10,17)"), "12");
    assert_eq!(r("modinv(2,4)"), "modinv(2, 4)"); // gcd(2,4)=2 ⇒ no inverse
    assert_eq!(r("jacobi(2,7)"), "1"); // 2 is a QR mod 7
    assert_eq!(r("jacobi(3,7)"), "-1");
    assert_eq!(r("jacobi(2,15)"), "1");
    assert_eq!(r("jacobi(6,9)"), "0"); // gcd(6,9) ≠ 1
}
#[test]
fn test_eval_number_theory_divisor_functions() {
    // Divisor functions: τ/numdivisors (count), σ/sigma (sum), and iscomposite (1/0). All exact via
    // integer factorization. σ(6) = 12 = 2·6 confirms the perfect number.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("tau(12)"), "6");
    assert_eq!(r("numdivisors(12)"), "6");
    assert_eq!(r("tau(7)"), "2"); // prime ⇒ 2 divisors
    assert_eq!(r("sigma(12)"), "28");
    assert_eq!(r("sigma(6)"), "12"); // perfect number: σ(n) = 2n
    assert_eq!(r("sigma(7)"), "8"); // prime p ⇒ σ = p + 1
    assert_eq!(r("iscomposite(12)"), "1");
    assert_eq!(r("iscomposite(7)"), "0"); // prime
    assert_eq!(r("iscomposite(1)"), "0"); // neither prime nor composite
}
#[test]
fn test_eval_number_theory_primes_and_totient() {
    // New number-theory functions: isprime (1/0, the engine has no boolean), nextprime, prevprime,
    // and Euler's totient. All exact (BigInt trial division / factorization).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("isprime(7)"), "1");
    assert_eq!(r("isprime(12)"), "0");
    assert_eq!(r("isprime(1)"), "0");
    assert_eq!(r("isprime(2)"), "1");
    assert_eq!(r("nextprime(10)"), "11");
    assert_eq!(r("nextprime(13)"), "17");
    assert_eq!(r("prevprime(10)"), "7");
    assert_eq!(r("prevprime(3)"), "2");
    // No prime below 2 ⇒ honest residual.
    assert_eq!(r("prevprime(2)"), "prevprime(2)");
    assert_eq!(r("totient(12)"), "4");
    assert_eq!(r("totient(7)"), "6"); // prime: φ(p) = p−1
    assert_eq!(r("totient(36)"), "12");
    // Controls: existing number-theory calls unchanged.
    assert_eq!(r("gcd(48,36)"), "12");
    assert_eq!(r("prime_factors(12)"), "2^2·3");
}
#[test]
fn test_eval_combinatorial_sequences() {
    // Combinatorial integer sequences: Fibonacci (F₀=0, F₁=1), Lucas (L₀=2, L₁=1), and Catalan
    // (Cₙ = (2n)!/((n+1)!·n!)), all computed by exact BigInt iteration. Negative indices decline
    // to honest residuals (the closed forms here are defined for n ≥ 0).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("fibonacci(10)"), "55");
    assert_eq!(r("fib(20)"), "6765");
    assert_eq!(r("fibonacci(0)"), "0");
    assert_eq!(r("fibonacci(1)"), "1");
    assert_eq!(r("lucas(10)"), "123");
    assert_eq!(r("lucas(0)"), "2");
    assert_eq!(r("catalan(5)"), "42");
    assert_eq!(r("catalan(0)"), "1");
    assert_eq!(r("catalan(10)"), "16796");
    // Negative index ⇒ honest residual.
    assert_eq!(r("fibonacci(-1)"), "fibonacci(-1)");
    assert_eq!(r("catalan(-2)"), "catalan(-2)");
}
#[test]
fn test_eval_bernoulli_and_stirling_numbers() {
    // Bernoulli numbers Bₙ (rational, B₁=−1/2 convention) and Stirling numbers of the second
    // (set partitions into k blocks) and first (unsigned: permutations with k cycles) kind. All
    // exact (BigInt/BigRational recurrences). Negative / k>n cases give honest residuals or 0.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("bernoulli(0)"), "1");
    assert_eq!(r("bernoulli(1)"), "-1/2");
    assert_eq!(r("bernoulli(2)"), "1/6");
    assert_eq!(r("bernoulli(3)"), "0"); // odd n>1 ⇒ 0
    assert_eq!(r("bernoulli(4)"), "-1/30");
    assert_eq!(r("bernoulli(6)"), "1/42");
    assert_eq!(r("bernoulli(-1)"), "bernoulli(-1)"); // honest residual
    assert_eq!(r("stirling2(4,2)"), "7");
    assert_eq!(r("stirling2(5,3)"), "25");
    assert_eq!(r("stirling2(0,0)"), "1");
    assert_eq!(r("stirling2(2,5)"), "0"); // k>n ⇒ 0
    assert_eq!(r("stirling1(4,2)"), "11"); // unsigned: permutations of 4 with 2 cycles
    assert_eq!(r("stirling1(5,2)"), "50");
    assert_eq!(r("stirling1(3,3)"), "1");
}
#[test]
fn test_eval_mixed_base_exponential_normalizes_to_common_prime() {
    // Terms with DIFFERENT integer bases that share a common prime (`4^x` and `2^x`, `9^x` and `3^x`)
    // used to error with "Cannot isolate: variable on both sides". They are now rewritten to the common
    // prime base (`4^x → 2^(2x)`), making the relation a polynomial in the single atom `p^x`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `u = 2^x`: `u² - 3u + 2 = 0 ⟹ u ∈ {1, 2} ⟹ x ∈ {0, 1}`.
    assert_eq!(r("solve(4^x - 3*2^x + 2 = 0, x)"), "{ 0, 1 }");
    assert_eq!(r("solve(9^x - 4*3^x + 3 = 0, x)"), "{ 0, 1 }");
    assert_eq!(r("solve(4^x - 5*2^x + 4 = 0, x)"), "{ 0, 2 }");
    // A branch out of range (`2^x = -1`) is dropped.
    assert_eq!(r("solve(4^x - 2^x - 2 = 0, x)"), "{ 1 }");
    // The inequality form normalizes too.
    assert_eq!(r("solve(4^x - 3*2^x + 2 < 0, x)"), "(0, 1)");
    // Three bases sharing the prime 2 (`8=2³, 4=2², 2=2¹`), a cubic in `2^x`.
    assert_eq!(r("solve(8^x - 6*4^x + 8*2^x = 0, x)"), "{ 1, 2 }");
    // Controls: a single base (already handled), base e (non-integer), and INCOMPATIBLE primes decline.
    assert_eq!(r("solve(2^(2*x) - 3*2^x + 2 = 0, x)"), "{ 0, 1 }");
    assert_eq!(r("solve(e^(2*x) - 3*e^x + 2 = 0, x)"), "{ ln(2), 0 }");
    assert_eq!(r("solve(2^x = 8, x)"), "{ 3 }");
}
#[test]
fn test_eval_derangement_isperfect_harmonic() {
    // derangement(n)/subfactorial (permutations with no fixed point), isperfect(n) (σ(n)=2n, 1/0 —
    // the engine has no boolean), and harmonic(n) = Σ_{k=1}^n 1/k (exact rational). All BigInt/
    // BigRational exact; isperfect reuses the same divisor-sum core as sigma.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("derangement(0)"), "1");
    assert_eq!(r("derangement(1)"), "0");
    assert_eq!(r("derangement(4)"), "9");
    assert_eq!(r("derangement(5)"), "44");
    assert_eq!(r("subfactorial(4)"), "9"); // alias
    assert_eq!(r("derangement(-1)"), "derangement(-1)"); // honest residual
    assert_eq!(r("isperfect(6)"), "1");
    assert_eq!(r("isperfect(28)"), "1");
    assert_eq!(r("isperfect(496)"), "1");
    assert_eq!(r("isperfect(12)"), "0");
    assert_eq!(r("isperfect(1)"), "0"); // 1 is not perfect (σ(1)=1)
    assert_eq!(r("harmonic(1)"), "1");
    assert_eq!(r("harmonic(4)"), "25/12");
    assert_eq!(r("harmonic(5)"), "137/60");
    // Control: sigma (which now shares the divisor-sum core) is unchanged.
    assert_eq!(r("sigma(28)"), "56");
}
