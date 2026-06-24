# Engine Improvement Scorecard

- Generated: 2026-06-24T10:56:18.355823+00:00
- Git branch: main
- Git commit: `5dc56fde6aafde5b6863b58d3ac96d5f3428fd63`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=790.07ms avg_case_ms=7.90 simplify=225.71ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=701.22ms avg_case_ms=3.51 simplify=238.18ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=483.25ms avg_case_ms=4.83 simplify=141.39ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=334.17ms avg_case_ms=6.68 simplify=106.94ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=238.18ms avg_simplify_ms=1.19 wall=701.22ms, shifted_quotient simplify=225.71ms avg_simplify_ms=2.26 wall=790.07ms, product simplify=141.39ms avg_simplify_ms=1.41 wall=483.25ms, difference simplify=106.94ms avg_simplify_ms=2.14 wall=334.17ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=790.07ms avg_case_ms=7.90 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=506.01ms avg_case_ms=5.06 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=483.25ms avg_case_ms=4.83 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=334.17ms avg_case_ms=6.68 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=195.21ms avg_case_ms=1.95 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.49ms median_wire=13.56ms median_wall=50.66ms, difference@0+50 #174 difference runs=3 median_simplify=12.21ms median_wire=12.26ms median_wall=47.40ms, product@0+100 #175 product runs=3 median_simplify=12.07ms median_wire=12.12ms median_wall=46.09ms, sum@0+100 #173 sum runs=3 median_simplify=12.38ms median_wire=12.43ms median_wall=46.57ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.91ms median_wire=10.99ms median_wall=41.46ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
