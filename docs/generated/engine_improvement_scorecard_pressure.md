# Engine Improvement Scorecard

- Generated: 2026-07-23T11:46:37.793271+00:00
- Git branch: main
- Git commit: `ec8e9f60a72f0c8ac3d80b11276f1640ca5b2061`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=10
- By area: calculus / integration:4, calculus / runtime:3, calculus / differentiation:2, calculus / robustness:1
- Recent 1: `calculus / integration` - 2026-06-08 - Discovery observe-only: polynomial cosecant/cotangent source-return still emits depth pressure
- Recent 2: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square atanh scaled-root runtime is not caused by the global empty-domain check
- Recent 3: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square inverse-root diff runtime is not fixed by raw target preservation

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=263
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.01s avg_case_ms=10.11 simplify=287.31ms avg_simplify_ms=2.87, sum total=200 failed=0 elapsed=895.68ms avg_case_ms=4.48 simplify=293.38ms avg_simplify_ms=1.47, product total=100 failed=0 elapsed=615.50ms avg_case_ms=6.16 simplify=178.60ms avg_simplify_ms=1.79, difference total=50 failed=0 elapsed=415.65ms avg_case_ms=8.31 simplify=127.43ms avg_simplify_ms=2.55
- Engine hotspots: sum simplify=293.38ms avg_simplify_ms=1.47 wall=895.68ms, shifted_quotient simplify=287.31ms avg_simplify_ms=2.87 wall=1.01s, product simplify=178.60ms avg_simplify_ms=1.79 wall=615.50ms, difference simplify=127.43ms avg_simplify_ms=2.55 wall=415.65ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.01s avg_case_ms=10.11 avg_simplify_ms=2.87, sum@0+100 failed=0 elapsed=660.73ms avg_case_ms=6.61 avg_simplify_ms=2.08, product@0+100 failed=0 elapsed=615.50ms avg_case_ms=6.16 avg_simplify_ms=1.79, difference@0+50 failed=0 elapsed=415.65ms avg_case_ms=8.31 avg_simplify_ms=2.55, sum@700+100 failed=0 elapsed=234.95ms avg_case_ms=2.35 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.92ms median_wire=17.00ms median_wall=65.35ms, difference@0+50 #174 difference runs=3 median_simplify=15.27ms median_wire=15.32ms median_wall=58.19ms, sum@0+100 #173 sum runs=3 median_simplify=15.00ms median_wire=15.05ms median_wall=58.10ms, product@0+100 #175 product runs=3 median_simplify=15.86ms median_wire=15.92ms median_wall=59.47ms, shifted_quotient@0+100 #112 shifted_quotient runs=3 median_simplify=9.96ms median_wire=10.02ms median_wall=37.06ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.94s | passed=450 failed=0 total=450 avg_case=6.533ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.73s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.95s | passed=1 failed=0 |
