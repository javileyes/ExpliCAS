# Engine Improvement Scorecard

- Generated: 2026-06-27T09:22:17.433581+00:00
- Git branch: main
- Git commit: `ff67720fdce7504d82090907812ed328e641a8c7`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=803.47ms avg_case_ms=8.03 simplify=231.61ms avg_simplify_ms=2.32, sum total=200 failed=0 elapsed=699.80ms avg_case_ms=3.50 simplify=241.85ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=483.69ms avg_case_ms=4.84 simplify=142.16ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=329.87ms avg_case_ms=6.60 simplify=106.28ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=241.85ms avg_simplify_ms=1.21 wall=699.80ms, shifted_quotient simplify=231.61ms avg_simplify_ms=2.32 wall=803.47ms, product simplify=142.16ms avg_simplify_ms=1.42 wall=483.69ms, difference simplify=106.28ms avg_simplify_ms=2.13 wall=329.87ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=803.47ms avg_case_ms=8.03 avg_simplify_ms=2.32, sum@0+100 failed=0 elapsed=504.43ms avg_case_ms=5.04 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=483.69ms avg_case_ms=4.84 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=329.87ms avg_case_ms=6.60 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=195.36ms avg_case_ms=1.95 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.23ms median_wire=13.32ms median_wall=50.63ms, product@0+100 #175 product runs=3 median_simplify=11.82ms median_wire=11.87ms median_wall=44.97ms, difference@0+50 #174 difference runs=3 median_simplify=11.89ms median_wire=11.95ms median_wall=45.13ms, sum@0+100 #173 sum runs=3 median_simplify=11.69ms median_wire=11.74ms median_wall=44.26ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.57ms median_wire=10.65ms median_wall=39.70ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
