# Engine Improvement Scorecard

- Generated: 2026-07-27T03:56:14.310956+00:00
- Git branch: main
- Git commit: `938f164954bf8129265258f68eeaedb0a66a6e91`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=930.70ms avg_case_ms=9.31 simplify=259.39ms avg_simplify_ms=2.59, sum total=200 failed=0 elapsed=821.34ms avg_case_ms=4.11 simplify=266.97ms avg_simplify_ms=1.33, product total=100 failed=0 elapsed=583.17ms avg_case_ms=5.83 simplify=167.02ms avg_simplify_ms=1.67, difference total=50 failed=0 elapsed=384.48ms avg_case_ms=7.69 simplify=116.74ms avg_simplify_ms=2.33
- Engine hotspots: sum simplify=266.97ms avg_simplify_ms=1.33 wall=821.34ms, shifted_quotient simplify=259.39ms avg_simplify_ms=2.59 wall=930.70ms, product simplify=167.02ms avg_simplify_ms=1.67 wall=583.17ms, difference simplify=116.74ms avg_simplify_ms=2.33 wall=384.48ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=930.70ms avg_case_ms=9.31 avg_simplify_ms=2.59, sum@0+100 failed=0 elapsed=599.05ms avg_case_ms=5.99 avg_simplify_ms=1.87, product@0+100 failed=0 elapsed=583.17ms avg_case_ms=5.83 avg_simplify_ms=1.67, difference@0+50 failed=0 elapsed=384.48ms avg_case_ms=7.69 avg_simplify_ms=2.33, sum@700+100 failed=0 elapsed=222.30ms avg_case_ms=2.22 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.25ms median_wire=16.31ms median_wall=61.44ms, difference@0+50 #174 difference runs=3 median_simplify=14.53ms median_wire=14.57ms median_wall=55.25ms, sum@0+100 #173 sum runs=3 median_simplify=14.29ms median_wire=14.34ms median_wall=55.40ms, product@0+100 #175 product runs=3 median_simplify=14.50ms median_wire=14.54ms median_wall=55.74ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.60ms median_wire=12.66ms median_wall=47.61ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.72s | passed=450 failed=0 total=450 avg_case=6.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 11.99s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.83s | passed=1 failed=0 |
