# Engine Improvement Scorecard

- Generated: 2026-07-30T15:59:05.328807+00:00
- Git branch: main
- Git commit: `bb346467b8aec007a8074e28d0440466e6dac16d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=962.67ms avg_case_ms=9.63 simplify=269.89ms avg_simplify_ms=2.70, sum total=200 failed=0 elapsed=845.07ms avg_case_ms=4.23 simplify=270.66ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=592.31ms avg_case_ms=5.92 simplify=170.59ms avg_simplify_ms=1.71, difference total=50 failed=0 elapsed=385.17ms avg_case_ms=7.70 simplify=113.01ms avg_simplify_ms=2.26
- Engine hotspots: sum simplify=270.66ms avg_simplify_ms=1.35 wall=845.07ms, shifted_quotient simplify=269.89ms avg_simplify_ms=2.70 wall=962.67ms, product simplify=170.59ms avg_simplify_ms=1.71 wall=592.31ms, difference simplify=113.01ms avg_simplify_ms=2.26 wall=385.17ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=962.67ms avg_case_ms=9.63 avg_simplify_ms=2.70, sum@0+100 failed=0 elapsed=615.60ms avg_case_ms=6.16 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=592.31ms avg_case_ms=5.92 avg_simplify_ms=1.71, difference@0+50 failed=0 elapsed=385.17ms avg_case_ms=7.70 avg_simplify_ms=2.26, sum@700+100 failed=0 elapsed=229.47ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.25ms median_wire=17.33ms median_wall=66.16ms, sum@0+100 #173 sum runs=3 median_simplify=16.43ms median_wire=16.48ms median_wall=62.20ms, product@0+100 #175 product runs=3 median_simplify=16.06ms median_wire=16.11ms median_wall=61.06ms, difference@0+50 #174 difference runs=3 median_simplify=15.23ms median_wire=15.28ms median_wall=57.78ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.92ms median_wire=12.99ms median_wall=48.82ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.79s | passed=450 failed=0 total=450 avg_case=6.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.09s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.73s | passed=1 failed=0 |
