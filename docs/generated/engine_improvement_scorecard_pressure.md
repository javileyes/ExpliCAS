# Engine Improvement Scorecard

- Generated: 2026-06-21T13:08:02.262389+00:00
- Git branch: main
- Git commit: `17e211bd6472cd4b8c7c0c1dbd4362ab18cb375e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=785.12ms avg_case_ms=7.85 simplify=224.33ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=700.46ms avg_case_ms=3.50 simplify=237.37ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=480.62ms avg_case_ms=4.81 simplify=137.92ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=330.29ms avg_case_ms=6.61 simplify=105.76ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=237.37ms avg_simplify_ms=1.19 wall=700.46ms, shifted_quotient simplify=224.33ms avg_simplify_ms=2.24 wall=785.12ms, product simplify=137.92ms avg_simplify_ms=1.38 wall=480.62ms, difference simplify=105.76ms avg_simplify_ms=2.12 wall=330.29ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=785.12ms avg_case_ms=7.85 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=505.55ms avg_case_ms=5.06 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=480.62ms avg_case_ms=4.81 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=330.29ms avg_case_ms=6.61 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=194.91ms avg_case_ms=1.95 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.99ms median_wire=13.06ms median_wall=49.41ms, sum@0+100 #173 sum runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.54ms, product@0+100 #175 product runs=3 median_simplify=11.79ms median_wire=11.84ms median_wall=44.60ms, difference@0+50 #174 difference runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.28ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.67ms median_wire=10.74ms median_wall=40.42ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
