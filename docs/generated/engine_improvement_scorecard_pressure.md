# Engine Improvement Scorecard

- Generated: 2026-06-28T13:41:30.768480+00:00
- Git branch: main
- Git commit: `9c8c8e974d7e0fb21773f483e9ab2759b5a55ab5`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=806.98ms avg_case_ms=8.07 simplify=231.65ms avg_simplify_ms=2.32, sum total=200 failed=0 elapsed=718.81ms avg_case_ms=3.59 simplify=248.52ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=490.47ms avg_case_ms=4.90 simplify=144.82ms avg_simplify_ms=1.45, difference total=50 failed=0 elapsed=339.34ms avg_case_ms=6.79 simplify=110.94ms avg_simplify_ms=2.22
- Engine hotspots: sum simplify=248.52ms avg_simplify_ms=1.24 wall=718.81ms, shifted_quotient simplify=231.65ms avg_simplify_ms=2.32 wall=806.98ms, product simplify=144.82ms avg_simplify_ms=1.45 wall=490.47ms, difference simplify=110.94ms avg_simplify_ms=2.22 wall=339.34ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=806.98ms avg_case_ms=8.07 avg_simplify_ms=2.32, sum@0+100 failed=0 elapsed=515.33ms avg_case_ms=5.15 avg_simplify_ms=1.70, product@0+100 failed=0 elapsed=490.47ms avg_case_ms=4.90 avg_simplify_ms=1.45, difference@0+50 failed=0 elapsed=339.34ms avg_case_ms=6.79 avg_simplify_ms=2.22, sum@700+100 failed=0 elapsed=203.48ms avg_case_ms=2.03 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.42ms median_wire=13.48ms median_wall=51.03ms, product@0+100 #175 product runs=3 median_simplify=11.81ms median_wire=11.86ms median_wall=45.13ms, difference@0+50 #174 difference runs=3 median_simplify=11.79ms median_wire=11.84ms median_wall=44.86ms, sum@0+100 #173 sum runs=3 median_simplify=11.69ms median_wire=11.74ms median_wall=44.73ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.95ms median_wire=11.03ms median_wall=41.69ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.36s | passed=450 failed=0 total=450 avg_case=5.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.52s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.11s | passed=1 failed=0 |
