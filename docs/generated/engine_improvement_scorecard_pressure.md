# Engine Improvement Scorecard

- Generated: 2026-07-30T20:16:09.480843+00:00
- Git branch: main
- Git commit: `6674f9a47fd38ae74622ae70838a33d58c500b65`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=965.50ms avg_case_ms=9.66 simplify=270.05ms avg_simplify_ms=2.70, sum total=200 failed=0 elapsed=847.94ms avg_case_ms=4.24 simplify=271.40ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=597.06ms avg_case_ms=5.97 simplify=171.71ms avg_simplify_ms=1.72, difference total=50 failed=0 elapsed=390.81ms avg_case_ms=7.82 simplify=114.86ms avg_simplify_ms=2.30
- Engine hotspots: sum simplify=271.40ms avg_simplify_ms=1.36 wall=847.94ms, shifted_quotient simplify=270.05ms avg_simplify_ms=2.70 wall=965.50ms, product simplify=171.71ms avg_simplify_ms=1.72 wall=597.06ms, difference simplify=114.86ms avg_simplify_ms=2.30 wall=390.81ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=965.50ms avg_case_ms=9.66 avg_simplify_ms=2.70, sum@0+100 failed=0 elapsed=616.95ms avg_case_ms=6.17 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=597.06ms avg_case_ms=5.97 avg_simplify_ms=1.72, difference@0+50 failed=0 elapsed=390.81ms avg_case_ms=7.82 avg_simplify_ms=2.30, sum@700+100 failed=0 elapsed=230.99ms avg_case_ms=2.31 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.49ms median_wire=16.56ms median_wall=62.81ms, difference@0+50 #174 difference runs=3 median_simplify=15.10ms median_wire=15.15ms median_wall=57.76ms, product@0+100 #175 product runs=3 median_simplify=14.96ms median_wire=15.01ms median_wall=56.83ms, sum@0+100 #173 sum runs=3 median_simplify=14.91ms median_wire=14.95ms median_wall=56.73ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.58ms median_wire=12.64ms median_wall=47.71ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.80s | passed=450 failed=0 total=450 avg_case=6.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.07s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.73s | passed=1 failed=0 |
