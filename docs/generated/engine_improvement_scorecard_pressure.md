# Engine Improvement Scorecard

- Generated: 2026-06-24T20:13:51.967506+00:00
- Git branch: main
- Git commit: `5208f4f18fbbb57a4d64c3b156a9696eb6723217`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=807.16ms avg_case_ms=8.07 simplify=231.98ms avg_simplify_ms=2.32, sum total=200 failed=0 elapsed=713.09ms avg_case_ms=3.57 simplify=242.55ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=489.36ms avg_case_ms=4.89 simplify=141.72ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=346.23ms avg_case_ms=6.92 simplify=110.94ms avg_simplify_ms=2.22
- Engine hotspots: sum simplify=242.55ms avg_simplify_ms=1.21 wall=713.09ms, shifted_quotient simplify=231.98ms avg_simplify_ms=2.32 wall=807.16ms, product simplify=141.72ms avg_simplify_ms=1.42 wall=489.36ms, difference simplify=110.94ms avg_simplify_ms=2.22 wall=346.23ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=807.16ms avg_case_ms=8.07 avg_simplify_ms=2.32, sum@0+100 failed=0 elapsed=513.39ms avg_case_ms=5.13 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=489.36ms avg_case_ms=4.89 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=346.23ms avg_case_ms=6.92 avg_simplify_ms=2.22, sum@700+100 failed=0 elapsed=199.70ms avg_case_ms=2.00 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.03ms median_wire=13.10ms median_wall=49.62ms, difference@0+50 #174 difference runs=3 median_simplify=12.40ms median_wire=12.45ms median_wall=46.52ms, sum@0+100 #173 sum runs=3 median_simplify=12.42ms median_wire=12.47ms median_wall=47.19ms, product@0+100 #175 product runs=3 median_simplify=11.79ms median_wire=11.84ms median_wall=44.63ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.16ms median_wire=11.23ms median_wall=42.01ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.36s | passed=450 failed=0 total=450 avg_case=5.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
