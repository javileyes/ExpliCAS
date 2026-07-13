# Engine Improvement Scorecard

- Generated: 2026-07-13T08:55:44.364934+00:00
- Git branch: main
- Git commit: `8ac43a7040e58d4162b8b061b48e7fdddd6a4377`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=966.29ms avg_case_ms=9.66 simplify=270.37ms avg_simplify_ms=2.70, sum total=200 failed=0 elapsed=845.53ms avg_case_ms=4.23 simplify=271.18ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=581.67ms avg_case_ms=5.82 simplify=165.16ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=385.62ms avg_case_ms=7.71 simplify=116.41ms avg_simplify_ms=2.33
- Engine hotspots: sum simplify=271.18ms avg_simplify_ms=1.36 wall=845.53ms, shifted_quotient simplify=270.37ms avg_simplify_ms=2.70 wall=966.29ms, product simplify=165.16ms avg_simplify_ms=1.65 wall=581.67ms, difference simplify=116.41ms avg_simplify_ms=2.33 wall=385.62ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=966.29ms avg_case_ms=9.66 avg_simplify_ms=2.70, sum@0+100 failed=0 elapsed=622.23ms avg_case_ms=6.22 avg_simplify_ms=1.91, product@0+100 failed=0 elapsed=581.67ms avg_case_ms=5.82 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=385.62ms avg_case_ms=7.71 avg_simplify_ms=2.33, sum@700+100 failed=0 elapsed=223.30ms avg_case_ms=2.23 avg_simplify_ms=0.80
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=14.81ms median_wire=14.85ms median_wall=56.04ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.42ms median_wire=16.48ms median_wall=62.36ms, product@0+100 #175 product runs=3 median_simplify=14.88ms median_wire=14.92ms median_wall=57.68ms, difference@0+50 #174 difference runs=3 median_simplify=15.41ms median_wire=15.46ms median_wall=59.23ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.80ms median_wire=12.88ms median_wall=48.22ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.78s | passed=450 failed=0 total=450 avg_case=6.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.01s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
