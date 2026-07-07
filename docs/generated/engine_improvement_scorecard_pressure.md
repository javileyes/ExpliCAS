# Engine Improvement Scorecard

- Generated: 2026-07-07T21:00:12.470548+00:00
- Git branch: main
- Git commit: `c4c0415cbad21f657e500bb6be717203a5c14781`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=917.38ms avg_case_ms=9.17 simplify=254.66ms avg_simplify_ms=2.55, sum total=200 failed=0 elapsed=824.64ms avg_case_ms=4.12 simplify=266.45ms avg_simplify_ms=1.33, product total=100 failed=0 elapsed=573.46ms avg_case_ms=5.73 simplify=163.11ms avg_simplify_ms=1.63, difference total=50 failed=0 elapsed=382.57ms avg_case_ms=7.65 simplify=116.51ms avg_simplify_ms=2.33
- Engine hotspots: sum simplify=266.45ms avg_simplify_ms=1.33 wall=824.64ms, shifted_quotient simplify=254.66ms avg_simplify_ms=2.55 wall=917.38ms, product simplify=163.11ms avg_simplify_ms=1.63 wall=573.46ms, difference simplify=116.51ms avg_simplify_ms=2.33 wall=382.57ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=917.38ms avg_case_ms=9.17 avg_simplify_ms=2.55, sum@0+100 failed=0 elapsed=600.41ms avg_case_ms=6.00 avg_simplify_ms=1.86, product@0+100 failed=0 elapsed=573.46ms avg_case_ms=5.73 avg_simplify_ms=1.63, difference@0+50 failed=0 elapsed=382.57ms avg_case_ms=7.65 avg_simplify_ms=2.33, sum@700+100 failed=0 elapsed=224.23ms avg_case_ms=2.24 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.21ms median_wire=16.28ms median_wall=64.37ms, sum@0+100 #173 sum runs=3 median_simplify=14.59ms median_wire=14.63ms median_wall=56.04ms, product@0+100 #175 product runs=3 median_simplify=14.48ms median_wire=14.52ms median_wall=56.13ms, difference@0+50 #174 difference runs=3 median_simplify=14.55ms median_wire=14.59ms median_wall=56.08ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.22ms median_wire=12.29ms median_wall=47.45ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.70s | passed=450 failed=0 total=450 avg_case=6.000ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
