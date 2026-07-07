# Engine Improvement Scorecard

- Generated: 2026-07-07T10:21:03.669893+00:00
- Git branch: main
- Git commit: `d284aaea96655081da269253c37d185dad2864fd`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=994.94ms avg_case_ms=9.95 simplify=274.17ms avg_simplify_ms=2.74, sum total=200 failed=0 elapsed=819.68ms avg_case_ms=4.10 simplify=264.93ms avg_simplify_ms=1.32, product total=100 failed=0 elapsed=581.45ms avg_case_ms=5.81 simplify=165.26ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=385.58ms avg_case_ms=7.71 simplify=115.70ms avg_simplify_ms=2.31
- Engine hotspots: shifted_quotient simplify=274.17ms avg_simplify_ms=2.74 wall=994.94ms, sum simplify=264.93ms avg_simplify_ms=1.32 wall=819.68ms, product simplify=165.26ms avg_simplify_ms=1.65 wall=581.45ms, difference simplify=115.70ms avg_simplify_ms=2.31 wall=385.58ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=994.94ms avg_case_ms=9.95 avg_simplify_ms=2.74, sum@0+100 failed=0 elapsed=596.32ms avg_case_ms=5.96 avg_simplify_ms=1.86, product@0+100 failed=0 elapsed=581.45ms avg_case_ms=5.81 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=385.58ms avg_case_ms=7.71 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=223.36ms avg_case_ms=2.23 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.38ms median_wire=16.45ms median_wall=63.24ms, sum@0+100 #173 sum runs=3 median_simplify=15.09ms median_wire=15.14ms median_wall=56.88ms, difference@0+50 #174 difference runs=3 median_simplify=14.86ms median_wire=14.91ms median_wall=56.90ms, product@0+100 #175 product runs=3 median_simplify=14.96ms median_wire=15.02ms median_wall=56.56ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.61ms median_wire=12.68ms median_wall=48.04ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.78s | passed=450 failed=0 total=450 avg_case=6.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
