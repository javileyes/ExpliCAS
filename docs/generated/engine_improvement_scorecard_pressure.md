# Engine Improvement Scorecard

- Generated: 2026-06-27T22:21:42.235016+00:00
- Git branch: main
- Git commit: `66524a1b5054a4b3af309d9248bf30485f15ecec`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=826.60ms avg_case_ms=8.27 simplify=237.81ms avg_simplify_ms=2.38, sum total=200 failed=0 elapsed=737.22ms avg_case_ms=3.69 simplify=259.12ms avg_simplify_ms=1.30, product total=100 failed=0 elapsed=561.04ms avg_case_ms=5.61 simplify=182.92ms avg_simplify_ms=1.83, difference total=50 failed=0 elapsed=346.35ms avg_case_ms=6.93 simplify=113.63ms avg_simplify_ms=2.27
- Engine hotspots: sum simplify=259.12ms avg_simplify_ms=1.30 wall=737.22ms, shifted_quotient simplify=237.81ms avg_simplify_ms=2.38 wall=826.60ms, product simplify=182.92ms avg_simplify_ms=1.83 wall=561.04ms, difference simplify=113.63ms avg_simplify_ms=2.27 wall=346.35ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=826.60ms avg_case_ms=8.27 avg_simplify_ms=2.38, product@0+100 failed=0 elapsed=561.04ms avg_case_ms=5.61 avg_simplify_ms=1.83, sum@0+100 failed=0 elapsed=531.46ms avg_case_ms=5.31 avg_simplify_ms=1.79, difference@0+50 failed=0 elapsed=346.35ms avg_case_ms=6.93 avg_simplify_ms=2.27, sum@700+100 failed=0 elapsed=205.76ms avg_case_ms=2.06 avg_simplify_ms=0.80
- Steady-state engine reruns: product@0+100 #175 product runs=3 median_simplify=12.10ms median_wire=12.15ms median_wall=45.95ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.35ms median_wire=13.44ms median_wall=50.81ms, difference@0+50 #174 difference runs=3 median_simplify=12.14ms median_wire=12.20ms median_wall=46.25ms, sum@0+100 #173 sum runs=3 median_simplify=11.97ms median_wire=12.02ms median_wall=45.97ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.11ms median_wire=11.19ms median_wall=41.64ms
- Steady-state dominant expressions: product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.47s | passed=450 failed=0 total=450 avg_case=5.489ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.63s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.12s | passed=1 failed=0 |
