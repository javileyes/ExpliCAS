# Engine Improvement Scorecard

- Generated: 2026-06-26T23:13:34.104923+00:00
- Git branch: main
- Git commit: `0bcace238a36f8eb909d081a7e69aef8691dd4bb`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=815.52ms avg_case_ms=8.16 simplify=235.63ms avg_simplify_ms=2.36, sum total=200 failed=0 elapsed=723.52ms avg_case_ms=3.62 simplify=250.10ms avg_simplify_ms=1.25, product total=100 failed=0 elapsed=513.89ms avg_case_ms=5.14 simplify=153.78ms avg_simplify_ms=1.54, difference total=50 failed=0 elapsed=343.81ms avg_case_ms=6.88 simplify=111.25ms avg_simplify_ms=2.22
- Engine hotspots: sum simplify=250.10ms avg_simplify_ms=1.25 wall=723.52ms, shifted_quotient simplify=235.63ms avg_simplify_ms=2.36 wall=815.52ms, product simplify=153.78ms avg_simplify_ms=1.54 wall=513.89ms, difference simplify=111.25ms avg_simplify_ms=2.22 wall=343.81ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=815.52ms avg_case_ms=8.16 avg_simplify_ms=2.36, sum@0+100 failed=0 elapsed=521.54ms avg_case_ms=5.22 avg_simplify_ms=1.72, product@0+100 failed=0 elapsed=513.89ms avg_case_ms=5.14 avg_simplify_ms=1.54, difference@0+50 failed=0 elapsed=343.81ms avg_case_ms=6.88 avg_simplify_ms=2.22, sum@700+100 failed=0 elapsed=201.98ms avg_case_ms=2.02 avg_simplify_ms=0.78
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.20ms median_wire=13.26ms median_wall=49.77ms, difference@0+50 #174 difference runs=3 median_simplify=11.88ms median_wire=11.93ms median_wall=47.51ms, product@0+100 #175 product runs=3 median_simplify=11.91ms median_wire=11.96ms median_wall=46.47ms, sum@0+100 #173 sum runs=3 median_simplify=13.12ms median_wire=13.18ms median_wall=48.61ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.85ms median_wire=10.92ms median_wall=43.95ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.40s | passed=450 failed=0 total=450 avg_case=5.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.09s | passed=1 failed=0 |
