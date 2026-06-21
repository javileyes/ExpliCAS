# Engine Improvement Scorecard

- Generated: 2026-06-21T17:31:22.226770+00:00
- Git branch: main
- Git commit: `ebb40acd39afe2c351ac04bc3c3a181e105faded`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=800.50ms avg_case_ms=8.01 simplify=230.36ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=723.05ms avg_case_ms=3.62 simplify=247.73ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=501.68ms avg_case_ms=5.02 simplify=145.66ms avg_simplify_ms=1.46, difference total=50 failed=0 elapsed=343.05ms avg_case_ms=6.86 simplify=109.57ms avg_simplify_ms=2.19
- Engine hotspots: sum simplify=247.73ms avg_simplify_ms=1.24 wall=723.05ms, shifted_quotient simplify=230.36ms avg_simplify_ms=2.30 wall=800.50ms, product simplify=145.66ms avg_simplify_ms=1.46 wall=501.68ms, difference simplify=109.57ms avg_simplify_ms=2.19 wall=343.05ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=800.50ms avg_case_ms=8.01 avg_simplify_ms=2.30, sum@0+100 failed=0 elapsed=523.33ms avg_case_ms=5.23 avg_simplify_ms=1.72, product@0+100 failed=0 elapsed=501.68ms avg_case_ms=5.02 avg_simplify_ms=1.46, difference@0+50 failed=0 elapsed=343.05ms avg_case_ms=6.86 avg_simplify_ms=2.19, sum@700+100 failed=0 elapsed=199.73ms avg_case_ms=2.00 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.61ms median_wire=13.69ms median_wall=51.44ms, sum@0+100 #173 sum runs=3 median_simplify=12.08ms median_wire=12.13ms median_wall=45.45ms, difference@0+50 #174 difference runs=3 median_simplify=11.87ms median_wire=11.92ms median_wall=45.13ms, product@0+100 #175 product runs=3 median_simplify=11.96ms median_wire=12.01ms median_wall=45.43ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.82ms median_wire=10.90ms median_wall=40.79ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.37s | passed=450 failed=0 total=450 avg_case=5.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.47s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
