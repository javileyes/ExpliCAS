# Engine Improvement Scorecard

- Generated: 2026-06-20T09:52:30.490160+00:00
- Git branch: main
- Git commit: `f526618af5df20f90c09c412ccf100a4dc87589e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=808.34ms avg_case_ms=8.08 simplify=232.47ms avg_simplify_ms=2.32, sum total=200 failed=0 elapsed=710.97ms avg_case_ms=3.55 simplify=244.24ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=495.75ms avg_case_ms=4.96 simplify=143.77ms avg_simplify_ms=1.44, difference total=50 failed=0 elapsed=339.92ms avg_case_ms=6.80 simplify=109.10ms avg_simplify_ms=2.18
- Engine hotspots: sum simplify=244.24ms avg_simplify_ms=1.22 wall=710.97ms, shifted_quotient simplify=232.47ms avg_simplify_ms=2.32 wall=808.34ms, product simplify=143.77ms avg_simplify_ms=1.44 wall=495.75ms, difference simplify=109.10ms avg_simplify_ms=2.18 wall=339.92ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=808.34ms avg_case_ms=8.08 avg_simplify_ms=2.32, sum@0+100 failed=0 elapsed=514.34ms avg_case_ms=5.14 avg_simplify_ms=1.70, product@0+100 failed=0 elapsed=495.75ms avg_case_ms=4.96 avg_simplify_ms=1.44, difference@0+50 failed=0 elapsed=339.92ms avg_case_ms=6.80 avg_simplify_ms=2.18, sum@700+100 failed=0 elapsed=196.63ms avg_case_ms=1.97 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.27ms median_wire=13.34ms median_wall=50.45ms, difference@0+50 #174 difference runs=3 median_simplify=11.77ms median_wire=11.81ms median_wall=45.00ms, product@0+100 #175 product runs=3 median_simplify=11.91ms median_wire=11.96ms median_wall=44.95ms, sum@0+100 #173 sum runs=3 median_simplify=11.70ms median_wire=11.76ms median_wall=44.25ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.91ms median_wire=10.98ms median_wall=40.86ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.36s | passed=450 failed=0 total=450 avg_case=5.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
