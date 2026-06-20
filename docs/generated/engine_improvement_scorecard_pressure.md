# Engine Improvement Scorecard

- Generated: 2026-06-20T17:20:23.307767+00:00
- Git branch: main
- Git commit: `9366d3923e7359b355a0f2b47d31798a1d397235`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=805.77ms avg_case_ms=8.06 simplify=230.88ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=707.89ms avg_case_ms=3.54 simplify=239.44ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=484.21ms avg_case_ms=4.84 simplify=140.15ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=334.97ms avg_case_ms=6.70 simplify=107.18ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=239.44ms avg_simplify_ms=1.20 wall=707.89ms, shifted_quotient simplify=230.88ms avg_simplify_ms=2.31 wall=805.77ms, product simplify=140.15ms avg_simplify_ms=1.40 wall=484.21ms, difference simplify=107.18ms avg_simplify_ms=2.14 wall=334.97ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=805.77ms avg_case_ms=8.06 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=513.51ms avg_case_ms=5.14 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=484.21ms avg_case_ms=4.84 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=334.97ms avg_case_ms=6.70 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=194.38ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.11ms median_wire=13.19ms median_wall=49.68ms, sum@0+100 #173 sum runs=3 median_simplify=11.95ms median_wire=12.01ms median_wall=45.04ms, difference@0+50 #174 difference runs=3 median_simplify=13.07ms median_wire=13.13ms median_wall=47.77ms, product@0+100 #175 product runs=3 median_simplify=11.85ms median_wire=11.90ms median_wall=44.89ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.76ms median_wire=10.84ms median_wall=40.60ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
