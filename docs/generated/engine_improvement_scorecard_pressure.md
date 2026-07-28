# Engine Improvement Scorecard

- Generated: 2026-07-28T23:13:39.928041+00:00
- Git branch: main
- Git commit: `21aaf610fb3046b2d437b83c1db55675ce7b4b6b`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=969.55ms avg_case_ms=9.70 simplify=270.80ms avg_simplify_ms=2.71, sum total=200 failed=0 elapsed=890.26ms avg_case_ms=4.45 simplify=290.37ms avg_simplify_ms=1.45, product total=100 failed=0 elapsed=620.11ms avg_case_ms=6.20 simplify=178.46ms avg_simplify_ms=1.78, difference total=50 failed=0 elapsed=404.29ms avg_case_ms=8.09 simplify=123.79ms avg_simplify_ms=2.48
- Engine hotspots: sum simplify=290.37ms avg_simplify_ms=1.45 wall=890.26ms, shifted_quotient simplify=270.80ms avg_simplify_ms=2.71 wall=969.55ms, product simplify=178.46ms avg_simplify_ms=1.78 wall=620.11ms, difference simplify=123.79ms avg_simplify_ms=2.48 wall=404.29ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=969.55ms avg_case_ms=9.70 avg_simplify_ms=2.71, sum@0+100 failed=0 elapsed=651.32ms avg_case_ms=6.51 avg_simplify_ms=2.05, product@0+100 failed=0 elapsed=620.11ms avg_case_ms=6.20 avg_simplify_ms=1.78, difference@0+50 failed=0 elapsed=404.29ms avg_case_ms=8.09 avg_simplify_ms=2.48, sum@700+100 failed=0 elapsed=238.94ms avg_case_ms=2.39 avg_simplify_ms=0.86
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.78ms median_wire=16.86ms median_wall=65.47ms, sum@0+100 #173 sum runs=3 median_simplify=15.73ms median_wire=15.79ms median_wall=60.03ms, difference@0+50 #174 difference runs=3 median_simplify=15.41ms median_wire=15.47ms median_wall=58.51ms, product@0+100 #175 product runs=3 median_simplify=15.36ms median_wire=15.41ms median_wall=58.15ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.05ms median_wire=13.12ms median_wall=49.16ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.89s | passed=450 failed=0 total=450 avg_case=6.422ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.37s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
