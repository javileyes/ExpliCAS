# Engine Improvement Scorecard

- Generated: 2026-06-15T18:57:30.618998+00:00
- Git branch: main
- Git commit: `321fbd06c94955fb86019b5b265a87f5a07fdbf9`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=788.44ms avg_case_ms=7.88 simplify=224.88ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=694.90ms avg_case_ms=3.47 simplify=232.76ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=476.63ms avg_case_ms=4.77 simplify=136.82ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=328.11ms avg_case_ms=6.56 simplify=104.33ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=232.76ms avg_simplify_ms=1.16 wall=694.90ms, shifted_quotient simplify=224.88ms avg_simplify_ms=2.25 wall=788.44ms, product simplify=136.82ms avg_simplify_ms=1.37 wall=476.63ms, difference simplify=104.33ms avg_simplify_ms=2.09 wall=328.11ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=788.44ms avg_case_ms=7.88 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=499.38ms avg_case_ms=4.99 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=476.63ms avg_case_ms=4.77 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=328.11ms avg_case_ms=6.56 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=195.52ms avg_case_ms=1.96 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.04ms median_wire=13.11ms median_wall=49.55ms, difference@0+50 #174 difference runs=3 median_simplify=11.72ms median_wire=11.77ms median_wall=44.35ms, product@0+100 #175 product runs=3 median_simplify=11.71ms median_wire=11.76ms median_wall=44.54ms, sum@0+100 #173 sum runs=3 median_simplify=11.62ms median_wire=11.67ms median_wall=44.38ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.69ms median_wire=10.76ms median_wall=40.47ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
