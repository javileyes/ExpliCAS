# Engine Improvement Scorecard

- Generated: 2026-06-26T08:08:39.917355+00:00
- Git branch: main
- Git commit: `42a5ab30fe383abf5713b64a3fcca494fb99d5d3`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=802.43ms avg_case_ms=8.02 simplify=230.68ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=708.01ms avg_case_ms=3.54 simplify=244.80ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=506.55ms avg_case_ms=5.07 simplify=150.39ms avg_simplify_ms=1.50, difference total=50 failed=0 elapsed=339.67ms avg_case_ms=6.79 simplify=108.85ms avg_simplify_ms=2.18
- Engine hotspots: sum simplify=244.80ms avg_simplify_ms=1.22 wall=708.01ms, shifted_quotient simplify=230.68ms avg_simplify_ms=2.31 wall=802.43ms, product simplify=150.39ms avg_simplify_ms=1.50 wall=506.55ms, difference simplify=108.85ms avg_simplify_ms=2.18 wall=339.67ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=802.43ms avg_case_ms=8.02 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=510.28ms avg_case_ms=5.10 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=506.55ms avg_case_ms=5.07 avg_simplify_ms=1.50, difference@0+50 failed=0 elapsed=339.67ms avg_case_ms=6.79 avg_simplify_ms=2.18, sum@700+100 failed=0 elapsed=197.73ms avg_case_ms=1.98 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.27ms median_wire=13.34ms median_wall=50.48ms, difference@0+50 #174 difference runs=3 median_simplify=11.89ms median_wire=11.94ms median_wall=45.14ms, product@0+100 #175 product runs=3 median_simplify=11.73ms median_wire=11.79ms median_wall=44.35ms, sum@0+100 #173 sum runs=3 median_simplify=11.57ms median_wire=11.62ms median_wall=44.17ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.53ms median_wire=10.61ms median_wall=40.48ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.36s | passed=450 failed=0 total=450 avg_case=5.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
