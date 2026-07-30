# Engine Improvement Scorecard

- Generated: 2026-07-30T08:05:03.459970+00:00
- Git branch: main
- Git commit: `7ec1c0388b323f8f53d72dbf0adbcb78b0d15b66`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=986.01ms avg_case_ms=9.86 simplify=279.90ms avg_simplify_ms=2.80, sum total=200 failed=0 elapsed=855.61ms avg_case_ms=4.28 simplify=276.58ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=609.77ms avg_case_ms=6.10 simplify=175.44ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=398.13ms avg_case_ms=7.96 simplify=117.22ms avg_simplify_ms=2.34
- Engine hotspots: shifted_quotient simplify=279.90ms avg_simplify_ms=2.80 wall=986.01ms, sum simplify=276.58ms avg_simplify_ms=1.38 wall=855.61ms, product simplify=175.44ms avg_simplify_ms=1.75 wall=609.77ms, difference simplify=117.22ms avg_simplify_ms=2.34 wall=398.13ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=986.01ms avg_case_ms=9.86 avg_simplify_ms=2.80, sum@0+100 failed=0 elapsed=625.19ms avg_case_ms=6.25 avg_simplify_ms=1.93, product@0+100 failed=0 elapsed=609.77ms avg_case_ms=6.10 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=398.13ms avg_case_ms=7.96 avg_simplify_ms=2.34, sum@700+100 failed=0 elapsed=230.42ms avg_case_ms=2.30 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.69ms median_wire=16.77ms median_wall=64.06ms, product@0+100 #175 product runs=3 median_simplify=15.19ms median_wire=15.25ms median_wall=58.18ms, sum@0+100 #173 sum runs=3 median_simplify=15.32ms median_wire=15.37ms median_wall=58.42ms, difference@0+50 #174 difference runs=3 median_simplify=15.26ms median_wire=15.32ms median_wall=58.13ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.08ms median_wire=13.15ms median_wall=49.67ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.85s | passed=450 failed=0 total=450 avg_case=6.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.37s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.75s | passed=1 failed=0 |
