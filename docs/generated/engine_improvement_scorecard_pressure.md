# Engine Improvement Scorecard

- Generated: 2026-07-30T15:25:48.221340+00:00
- Git branch: main
- Git commit: `b02d27bad257b9c9b08076ebee0e01f3c961864e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=983.55ms avg_case_ms=9.84 simplify=277.85ms avg_simplify_ms=2.78, sum total=200 failed=0 elapsed=847.11ms avg_case_ms=4.24 simplify=272.11ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=614.69ms avg_case_ms=6.15 simplify=176.06ms avg_simplify_ms=1.76, difference total=50 failed=0 elapsed=397.74ms avg_case_ms=7.95 simplify=116.69ms avg_simplify_ms=2.33
- Engine hotspots: shifted_quotient simplify=277.85ms avg_simplify_ms=2.78 wall=983.55ms, sum simplify=272.11ms avg_simplify_ms=1.36 wall=847.11ms, product simplify=176.06ms avg_simplify_ms=1.76 wall=614.69ms, difference simplify=116.69ms avg_simplify_ms=2.33 wall=397.74ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=983.55ms avg_case_ms=9.84 avg_simplify_ms=2.78, product@0+100 failed=0 elapsed=614.69ms avg_case_ms=6.15 avg_simplify_ms=1.76, sum@0+100 failed=0 elapsed=611.67ms avg_case_ms=6.12 avg_simplify_ms=1.87, difference@0+50 failed=0 elapsed=397.74ms avg_case_ms=7.95 avg_simplify_ms=2.33, sum@700+100 failed=0 elapsed=235.43ms avg_case_ms=2.35 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.83ms median_wire=16.91ms median_wall=64.40ms, difference@0+50 #174 difference runs=3 median_simplify=15.22ms median_wire=15.27ms median_wall=58.30ms, sum@0+100 #173 sum runs=3 median_simplify=15.41ms median_wire=15.46ms median_wall=58.73ms, product@0+100 #175 product runs=3 median_simplify=15.08ms median_wire=15.13ms median_wall=57.45ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.62ms median_wire=12.69ms median_wall=48.51ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.85s | passed=450 failed=0 total=450 avg_case=6.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.10s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.73s | passed=1 failed=0 |
