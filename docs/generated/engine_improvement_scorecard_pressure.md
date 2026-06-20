# Engine Improvement Scorecard

- Generated: 2026-06-20T05:22:08.590027+00:00
- Git branch: main
- Git commit: `785a2795160cd8964ed24b6e73159ca2b1d8ad42`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=791.01ms avg_case_ms=7.91 simplify=226.84ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=699.43ms avg_case_ms=3.50 simplify=237.53ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=499.02ms avg_case_ms=4.99 simplify=144.23ms avg_simplify_ms=1.44, difference total=50 failed=0 elapsed=343.57ms avg_case_ms=6.87 simplify=110.17ms avg_simplify_ms=2.20
- Engine hotspots: sum simplify=237.53ms avg_simplify_ms=1.19 wall=699.43ms, shifted_quotient simplify=226.84ms avg_simplify_ms=2.27 wall=791.01ms, product simplify=144.23ms avg_simplify_ms=1.44 wall=499.02ms, difference simplify=110.17ms avg_simplify_ms=2.20 wall=343.57ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=791.01ms avg_case_ms=7.91 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=507.34ms avg_case_ms=5.07 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=499.02ms avg_case_ms=4.99 avg_simplify_ms=1.44, difference@0+50 failed=0 elapsed=343.57ms avg_case_ms=6.87 avg_simplify_ms=2.20, sum@700+100 failed=0 elapsed=192.10ms avg_case_ms=1.92 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.21ms median_wire=13.28ms median_wall=50.14ms, product@0+100 #175 product runs=3 median_simplify=11.67ms median_wire=11.72ms median_wall=44.91ms, difference@0+50 #174 difference runs=3 median_simplify=13.49ms median_wire=13.55ms median_wall=51.47ms, sum@0+100 #173 sum runs=3 median_simplify=13.38ms median_wire=13.44ms median_wall=49.63ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.11ms median_wire=12.20ms median_wall=44.20ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.61s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
