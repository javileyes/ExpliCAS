# Engine Improvement Scorecard

- Generated: 2026-07-19T08:56:22.305733+00:00
- Git branch: main
- Git commit: `c599a6632b5eba2cffca8df379ae0887c8c4f254`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=970.72ms avg_case_ms=9.71 simplify=271.43ms avg_simplify_ms=2.71, sum total=200 failed=0 elapsed=878.76ms avg_case_ms=4.39 simplify=282.06ms avg_simplify_ms=1.41, product total=100 failed=0 elapsed=605.17ms avg_case_ms=6.05 simplify=174.50ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=397.20ms avg_case_ms=7.94 simplify=120.50ms avg_simplify_ms=2.41
- Engine hotspots: sum simplify=282.06ms avg_simplify_ms=1.41 wall=878.76ms, shifted_quotient simplify=271.43ms avg_simplify_ms=2.71 wall=970.72ms, product simplify=174.50ms avg_simplify_ms=1.74 wall=605.17ms, difference simplify=120.50ms avg_simplify_ms=2.41 wall=397.20ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=970.72ms avg_case_ms=9.71 avg_simplify_ms=2.71, sum@0+100 failed=0 elapsed=650.22ms avg_case_ms=6.50 avg_simplify_ms=2.00, product@0+100 failed=0 elapsed=605.17ms avg_case_ms=6.05 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=397.20ms avg_case_ms=7.94 avg_simplify_ms=2.41, sum@700+100 failed=0 elapsed=228.54ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.09ms median_wire=15.14ms median_wall=57.27ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.81ms median_wire=16.88ms median_wall=64.13ms, product@0+100 #175 product runs=3 median_simplify=15.30ms median_wire=15.36ms median_wall=66.94ms, difference@0+50 #174 difference runs=3 median_simplify=15.67ms median_wire=15.72ms median_wall=58.68ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.86ms median_wire=12.93ms median_wall=48.48ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.85s | passed=450 failed=0 total=450 avg_case=6.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
