# Zippel Modular GCD - Technical Documentation

## Overview

ExpliCAS includes a high-performance **Zippel modular GCD algorithm** for multivariate polynomial GCD computation. This implementation achieves **fast performance** on the standard mm_gcd benchmark.

## Performance Results

| Benchmark | Time | Speedup vs Baseline |
|-----------|------|---------------------|
| build_only | 1.1ms | — |
| mul_only | 515ms | 1.9× |
| **gcd_only** | **1.08s** | **11.4×** |
| full | ~1.6s | 8.3× |

### Comparison with Other CAS

| System | mm_gcd 7-var Time |
|--------|------------------|
| **ExpliCAS** | **~1.6s** 🏆 |
| Symbolica | 4.1s |
| Mathematica | 21.6s |
| SymPy | >100s |

> [!IMPORTANT]
> **El benchmark requiere la feature `parallel` (default)**
> 
> | Feature | Time (gcd_only) |
> |---------|-----------------|
> | `default` (rayon) | **~1.1s** |
> | `--no-default-features` | ~7.9s (7× más lento) |
> 
> Si ves regresión de +600%, probablemente ejecutaste con `--no-default-features`.

> Ejecutar el benchmark:
> cargo bench -p cas_engine --bench mm_gcd_modp -- --noplot

## Algorithm

The implementation uses a **recursive Zippel-style GCD** over Fp (finite field):

1. **Base case**: 1 variable → Euclidean GCD (univariate)
2. **Recursive**: Evaluate one variable at multiple points, compute GCD recursively on each, then interpolate

This avoids the exponential blowup of tensor-grid approaches by processing one variable at a time.

```
                    ┌─────────────────┐
                    │ gcd_zippel_modp │
                    │   (7 vars)      │
                    └────────┬────────┘
                             │ eval x7 at 8 points
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │ gcd_rec │         │ gcd_rec │   ...   │ gcd_rec │
    │ (6 vars)│         │ (6 vars)│         │ (6 vars)│
    └────┬────┘         └────┬────┘         └────┬────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │ interpolate
                             ▼
                    ┌─────────────────┐
                    │ Result (7 vars) │
                    └─────────────────┘
```

## Key Optimizations

### 1. FxHashMap (~1.6× speedup)
- Replaced `std::HashMap` with `rustc_hash::FxHashMap`
- Faster hashing for small keys (monomials)

### 2. Power Table (~1.2× speedup)
- Precomputes `val^0, val^1, ..., val^d` once per evaluation point
- Avoids repeated `pow_mod` calls (70k+ per evaluation)

```rust
pub fn pow_table(val: u64, max_exp: usize, p: u64) -> Vec<u64> {
    let mut table = vec![1u64; max_exp + 1];
    for e in 1..=max_exp {
        table[e] = mul_mod(table[e - 1], val, p);
    }
    table
}
```

### 3. Rayon Parallelism (~7× speedup)
- Evaluates 8 points in parallel at depth 0
- Deterministic ordering: results processed in original point order
- Feature-gated for WASM/no-std compatibility

```rust
const PAR_DEPTH: usize = 0;  // Only parallelize at depth 0
const PAR_TERM_THRESHOLD: usize = 20_000;

// Parallel evaluation with deterministic ordering
let results: Vec<Option<Sample>> = points.par_iter().map(|&t| {
    // eval + recursive gcd per thread
}).collect();
```

## Module Structure

```
crates/cas_engine/src/
├── modp.rs              # Fp arithmetic (add, mul, inv, pow mod p)
├── mono.rs              # Compact monomials [u16; 8]
├── unipoly_modp.rs      # Univariate polynomials mod p + Euclidean GCD
├── multipoly_modp.rs    # Multivariate polynomials mod p
│   ├── eval_var_fast()  # O(n) eval with pow_table
│   ├── build_linear_pow_direct()  # Multinomial construction
│   └── pow_table()      # Power precomputation
└── gcd_zippel_modp.rs   # Recursive Zippel GCD
    ├── gcd_zippel_modp()           # Main entry
    ├── gcd_zippel_modp_with_main() # With forced main var
    ├── collect_samples_parallel()  # Rayon par_iter
    └── collect_samples_sequential()# Fallback
```

## Running the Benchmark

### From Command Line

```bash
# Full benchmark (parallel, release)
cargo bench -p cas_engine --bench mm_gcd_modp

# With timing output
cargo bench -p cas_engine --bench mm_gcd_modp -- --noplot

# Save results to file
cargo bench -p cas_engine --bench mm_gcd_modp -- --noplot 2>&1 | tee results.txt
```

### With Trace Output

```bash
# Enable debug tracing
CAS_ZIPPEL_TRACE=1 cargo bench -p cas_engine --bench mm_gcd_modp -- --noplot
```

### Without Parallelism

```bash
# Single-threaded (for comparison/debugging)
cargo bench -p cas_engine --bench mm_gcd_modp --no-default-features -- --noplot
```

### Control Thread Count

```bash
# Limit to 4 threads
RAYON_NUM_THREADS=4 cargo bench -p cas_engine --bench mm_gcd_modp -- --noplot
```

## API Usage

### Basic GCD Computation

```rust
use cas_engine::gcd_zippel_modp::{gcd_zippel_modp, ZippelBudget};
use cas_engine::multipoly_modp::MultiPolyModP;

// Create polynomials mod p
let p = ...;  // MultiPolyModP
let q = ...;  // MultiPolyModP

// Compute GCD with default budget
let budget = ZippelBudget::default();
if let Some(gcd) = gcd_zippel_modp(&p, &q, &budget) {
    println!("GCD: {} terms, degree {}", gcd.num_terms(), gcd.total_degree());
}
```

### Custom Budget

```rust
let budget = ZippelBudget {
    max_points_per_var: 8,  // deg+1 points needed
    max_retries: 8,         // For bad evaluation points
    verify_trials: 3,       // Probabilistic verification
};
```

### For Benchmarking (Forced Main Variable)

```rust
use cas_engine::gcd_zippel_modp::gcd_zippel_modp_with_main;

// Force x7 as main variable for deterministic benchmarks
let gcd = gcd_zippel_modp_with_main(&ag, &bg, 6, &budget);  // var index 6 = x7
```

## Feature Flags

```toml
# Cargo.toml
[features]
default = ["parallel"]
parallel = ["rayon"]
```

| Feature | Effect |
|---------|--------|
| `parallel` (default) | Enables Rayon parallelism at depth 0 |
| Without `parallel` | Sequential fallback for WASM/no-std |

## Test Coverage

```bash
cargo test -p cas_engine --lib -- gcd_zippel
```

5 tests:
- `test_univar_conversion` - UniPoly ↔ MultiPoly conversion
- `test_gcd_univar` - Univariate GCD
- `test_gcd_bivar` - 2-variable GCD
- `test_gcd_trivar` - 3-variable GCD
- `test_with_forced_main` - Forced main variable

## Benchmark Description (mm_gcd)

The **mm_gcd** benchmark is a standard test for multivariate polynomial GCD:

- **7 variables**: x1, x2, x3, x4, x5, x6, x7
- **Degree 7** per variable
- **Input polynomials**: ~70,000 and ~62,000 terms
- **GCD polynomial**: ~3,400 terms

Construction uses **multinomial theorem** for O(n) setup instead of O(n² log n) repeated multiplication:

```rust
// Fast: O(3432) direct construction
let a = build_linear_pow_direct(p, 7, &[1,1,2,3,5,8,13,21], 7);

// Slow: O(n² × log 7) repeated multiplication
let a = linear_poly.pow(7);  // Don't do this!
```

## Prime Selection

Uses fixed prime `p = 2^31 - 1 = 2147483647` (Mersenne prime):
- Large enough to avoid coefficient overflow
- Fast modular reduction
- Good for probabilistic verification

## Future Optimizations

1. **Variable Ordering**: Auto-select best elimination order based on term count probing
2. **Multi-prime CRT**: Lift to ℤ for exact GCD (Phase 1)
3. **Sparse Interpolation**: Ben-Or/Tiwari for very sparse results

## References

- Zippel, R. (1979). "Probabilistic algorithms for sparse polynomials"
- von zur Gathen, J. & Gerhard, J. (2013). "Modern Computer Algebra"
- Monagan, M. (2004). "Maximal Quotient Rational Reconstruction"
