# Engine Improvement Scorecard

- Generated: 2026-06-17T06:41:11.589069+00:00
- Git branch: main
- Git commit: `afadce5ec16d453a1cb8686151d59143770c66c5`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=969.47ms avg_case_ms=9.69 simplify=246.49ms avg_simplify_ms=2.46, sum total=200 failed=0 elapsed=691.86ms avg_case_ms=3.46 simplify=233.53ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=450.89ms avg_case_ms=4.51 simplify=129.36ms avg_simplify_ms=1.29, difference total=50 failed=0 elapsed=304.11ms avg_case_ms=6.08 simplify=97.69ms avg_simplify_ms=1.95
- Engine hotspots: shifted_quotient simplify=246.49ms avg_simplify_ms=2.46 wall=969.47ms, sum simplify=233.53ms avg_simplify_ms=1.17 wall=691.86ms, product simplify=129.36ms avg_simplify_ms=1.29 wall=450.89ms, difference simplify=97.69ms avg_simplify_ms=1.95 wall=304.11ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=969.47ms avg_case_ms=9.69 avg_simplify_ms=2.46, sum@0+100 failed=0 elapsed=513.06ms avg_case_ms=5.13 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=450.89ms avg_case_ms=4.51 avg_simplify_ms=1.29, difference@0+50 failed=0 elapsed=304.11ms avg_case_ms=6.08 avg_simplify_ms=1.95, sum@700+100 failed=0 elapsed=178.80ms avg_case_ms=1.79 avg_simplify_ms=0.67
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.33ms median_wire=16.43ms median_wall=61.77ms, sum@0+100 #173 sum runs=3 median_simplify=15.29ms median_wire=15.36ms median_wall=58.63ms, difference@0+50 #174 difference runs=3 median_simplify=15.66ms median_wire=15.73ms median_wall=58.77ms, product@0+100 #175 product runs=3 median_simplify=15.09ms median_wire=15.16ms median_wall=57.10ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.07ms median_wire=13.17ms median_wall=49.38ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.42s | passed=450 failed=0 total=450 avg_case=5.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.94s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 874.99s | passed=1 failed=0 |
