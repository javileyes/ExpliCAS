# Engine Improvement Scorecard

- Generated: 2026-06-24T08:54:06.318000+00:00
- Git branch: main
- Git commit: `4506d7348d8da02d55c03211d720982ccab1e4aa`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=801.42ms avg_case_ms=8.01 simplify=229.52ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=722.44ms avg_case_ms=3.61 simplify=245.46ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=482.23ms avg_case_ms=4.82 simplify=139.48ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=335.68ms avg_case_ms=6.71 simplify=107.16ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=245.46ms avg_simplify_ms=1.23 wall=722.44ms, shifted_quotient simplify=229.52ms avg_simplify_ms=2.30 wall=801.42ms, product simplify=139.48ms avg_simplify_ms=1.39 wall=482.23ms, difference simplify=107.16ms avg_simplify_ms=2.14 wall=335.68ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=801.42ms avg_case_ms=8.01 avg_simplify_ms=2.30, sum@0+100 failed=0 elapsed=522.98ms avg_case_ms=5.23 avg_simplify_ms=1.71, product@0+100 failed=0 elapsed=482.23ms avg_case_ms=4.82 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=335.68ms avg_case_ms=6.71 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=199.46ms avg_case_ms=1.99 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.95ms median_wire=13.02ms median_wall=50.16ms, sum@0+100 #173 sum runs=3 median_simplify=12.32ms median_wire=12.38ms median_wall=47.63ms, product@0+100 #175 product runs=3 median_simplify=12.10ms median_wire=12.16ms median_wall=45.00ms, difference@0+50 #174 difference runs=3 median_simplify=11.73ms median_wire=11.78ms median_wall=44.43ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.36ms median_wire=10.44ms median_wall=39.78ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
