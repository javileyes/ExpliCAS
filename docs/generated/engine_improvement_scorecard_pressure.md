# Engine Improvement Scorecard

- Generated: 2026-07-17T01:37:59.925626+00:00
- Git branch: main
- Git commit: `ac9dbd0daecda8b62d5b298fe34e5f8816eb5c03`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.01s avg_case_ms=10.10 simplify=282.30ms avg_simplify_ms=2.82, sum total=200 failed=0 elapsed=896.11ms avg_case_ms=4.48 simplify=287.82ms avg_simplify_ms=1.44, product total=100 failed=0 elapsed=611.22ms avg_case_ms=6.11 simplify=175.10ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=410.33ms avg_case_ms=8.21 simplify=124.88ms avg_simplify_ms=2.50
- Engine hotspots: sum simplify=287.82ms avg_simplify_ms=1.44 wall=896.11ms, shifted_quotient simplify=282.30ms avg_simplify_ms=2.82 wall=1.01s, product simplify=175.10ms avg_simplify_ms=1.75 wall=611.22ms, difference simplify=124.88ms avg_simplify_ms=2.50 wall=410.33ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.01s avg_case_ms=10.10 avg_simplify_ms=2.82, sum@0+100 failed=0 elapsed=658.16ms avg_case_ms=6.58 avg_simplify_ms=2.02, product@0+100 failed=0 elapsed=611.22ms avg_case_ms=6.11 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=410.33ms avg_case_ms=8.21 avg_simplify_ms=2.50, sum@700+100 failed=0 elapsed=237.95ms avg_case_ms=2.38 avg_simplify_ms=0.86
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.73ms median_wire=16.80ms median_wall=64.60ms, difference@0+50 #174 difference runs=3 median_simplify=15.21ms median_wire=15.26ms median_wall=57.88ms, sum@0+100 #173 sum runs=3 median_simplify=15.16ms median_wire=15.22ms median_wall=58.30ms, product@0+100 #175 product runs=3 median_simplify=15.06ms median_wire=15.12ms median_wall=58.56ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.27ms median_wire=13.33ms median_wall=50.19ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.93s | passed=450 failed=0 total=450 avg_case=6.511ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.57s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
