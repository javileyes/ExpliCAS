# Engine Improvement Scorecard

- Generated: 2026-06-25T10:39:25.202765+00:00
- Git branch: main
- Git commit: `1e41faa35a67440f498e002f3dfd81934fef0a5a`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=782.96ms avg_case_ms=7.83 simplify=223.50ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=700.55ms avg_case_ms=3.50 simplify=238.11ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=474.81ms avg_case_ms=4.75 simplify=137.02ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=331.26ms avg_case_ms=6.63 simplify=106.07ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=238.11ms avg_simplify_ms=1.19 wall=700.55ms, shifted_quotient simplify=223.50ms avg_simplify_ms=2.24 wall=782.96ms, product simplify=137.02ms avg_simplify_ms=1.37 wall=474.81ms, difference simplify=106.07ms avg_simplify_ms=2.12 wall=331.26ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=782.96ms avg_case_ms=7.83 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=506.10ms avg_case_ms=5.06 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=474.81ms avg_case_ms=4.75 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=331.26ms avg_case_ms=6.63 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=194.45ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.10ms median_wire=13.17ms median_wall=49.67ms, difference@0+50 #174 difference runs=3 median_simplify=11.73ms median_wire=11.78ms median_wall=44.82ms, sum@0+100 #173 sum runs=3 median_simplify=11.88ms median_wire=11.93ms median_wall=45.03ms, product@0+100 #175 product runs=3 median_simplify=11.76ms median_wire=11.80ms median_wall=44.27ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.71ms median_wire=10.78ms median_wall=40.56ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
