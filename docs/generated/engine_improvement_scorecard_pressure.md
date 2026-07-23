# Engine Improvement Scorecard

- Generated: 2026-07-23T02:02:48.574887+00:00
- Git branch: main
- Git commit: `9e581125bec8f69bbb6034ddcd6054c0e4325415`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=975.43ms avg_case_ms=9.75 simplify=272.19ms avg_simplify_ms=2.72, sum total=200 failed=0 elapsed=882.67ms avg_case_ms=4.41 simplify=284.16ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=603.15ms avg_case_ms=6.03 simplify=173.34ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=400.09ms avg_case_ms=8.00 simplify=121.95ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=284.16ms avg_simplify_ms=1.42 wall=882.67ms, shifted_quotient simplify=272.19ms avg_simplify_ms=2.72 wall=975.43ms, product simplify=173.34ms avg_simplify_ms=1.73 wall=603.15ms, difference simplify=121.95ms avg_simplify_ms=2.44 wall=400.09ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=975.43ms avg_case_ms=9.75 avg_simplify_ms=2.72, sum@0+100 failed=0 elapsed=647.46ms avg_case_ms=6.47 avg_simplify_ms=2.00, product@0+100 failed=0 elapsed=603.15ms avg_case_ms=6.03 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=400.09ms avg_case_ms=8.00 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=235.21ms avg_case_ms=2.35 avg_simplify_ms=0.84
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.26ms median_wire=15.31ms median_wall=57.88ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.67ms median_wire=16.74ms median_wall=63.82ms, difference@0+50 #174 difference runs=3 median_simplify=15.29ms median_wire=15.34ms median_wall=58.33ms, product@0+100 #175 product runs=3 median_simplify=16.21ms median_wire=16.27ms median_wall=59.94ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.66ms median_wire=12.73ms median_wall=49.23ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.52s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.94s | passed=1 failed=0 |
