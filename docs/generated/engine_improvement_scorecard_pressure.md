# Engine Improvement Scorecard

- Generated: 2026-07-24T21:31:50.239088+00:00
- Git branch: main
- Git commit: `f7c8059a1424367a1c110acbfd30bf44e0b36d7d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.02s avg_case_ms=10.19 simplify=289.13ms avg_simplify_ms=2.89, sum total=200 failed=0 elapsed=900.50ms avg_case_ms=4.50 simplify=295.74ms avg_simplify_ms=1.48, product total=100 failed=0 elapsed=634.59ms avg_case_ms=6.35 simplify=182.55ms avg_simplify_ms=1.83, difference total=50 failed=0 elapsed=413.19ms avg_case_ms=8.26 simplify=126.72ms avg_simplify_ms=2.53
- Engine hotspots: sum simplify=295.74ms avg_simplify_ms=1.48 wall=900.50ms, shifted_quotient simplify=289.13ms avg_simplify_ms=2.89 wall=1.02s, product simplify=182.55ms avg_simplify_ms=1.83 wall=634.59ms, difference simplify=126.72ms avg_simplify_ms=2.53 wall=413.19ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.02s avg_case_ms=10.19 avg_simplify_ms=2.89, sum@0+100 failed=0 elapsed=660.76ms avg_case_ms=6.61 avg_simplify_ms=2.09, product@0+100 failed=0 elapsed=634.59ms avg_case_ms=6.35 avg_simplify_ms=1.83, difference@0+50 failed=0 elapsed=413.19ms avg_case_ms=8.26 avg_simplify_ms=2.53, sum@700+100 failed=0 elapsed=239.74ms avg_case_ms=2.40 avg_simplify_ms=0.87
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.94ms median_wire=17.01ms median_wall=65.73ms, product@0+100 #175 product runs=3 median_simplify=15.79ms median_wire=15.84ms median_wall=60.94ms, difference@0+50 #174 difference runs=3 median_simplify=15.74ms median_wire=15.79ms median_wall=60.37ms, sum@0+100 #173 sum runs=3 median_simplify=15.76ms median_wire=15.81ms median_wall=60.17ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.32ms median_wire=13.40ms median_wall=50.67ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.97s | passed=450 failed=0 total=450 avg_case=6.600ms |
| `calculus_diff_exhaustive_contract` | `pass` | 13.04s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
