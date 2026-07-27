# Engine Improvement Scorecard

- Generated: 2026-07-27T03:26:37.537545+00:00
- Git branch: main
- Git commit: `f856cf4b7e71826f9fe785b3e1bfdb9fbaa920aa`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=945.42ms avg_case_ms=9.45 simplify=264.29ms avg_simplify_ms=2.64, sum total=200 failed=0 elapsed=838.12ms avg_case_ms=4.19 simplify=271.82ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=590.45ms avg_case_ms=5.90 simplify=168.83ms avg_simplify_ms=1.69, difference total=50 failed=0 elapsed=389.06ms avg_case_ms=7.78 simplify=118.16ms avg_simplify_ms=2.36
- Engine hotspots: sum simplify=271.82ms avg_simplify_ms=1.36 wall=838.12ms, shifted_quotient simplify=264.29ms avg_simplify_ms=2.64 wall=945.42ms, product simplify=168.83ms avg_simplify_ms=1.69 wall=590.45ms, difference simplify=118.16ms avg_simplify_ms=2.36 wall=389.06ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=945.42ms avg_case_ms=9.45 avg_simplify_ms=2.64, sum@0+100 failed=0 elapsed=613.60ms avg_case_ms=6.14 avg_simplify_ms=1.92, product@0+100 failed=0 elapsed=590.45ms avg_case_ms=5.90 avg_simplify_ms=1.69, difference@0+50 failed=0 elapsed=389.06ms avg_case_ms=7.78 avg_simplify_ms=2.36, sum@700+100 failed=0 elapsed=224.52ms avg_case_ms=2.25 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.23ms median_wire=16.30ms median_wall=62.21ms, sum@0+100 #173 sum runs=3 median_simplify=14.77ms median_wire=14.82ms median_wall=55.93ms, difference@0+50 #174 difference runs=3 median_simplify=14.90ms median_wire=14.94ms median_wall=56.65ms, product@0+100 #175 product runs=3 median_simplify=14.68ms median_wire=14.72ms median_wall=56.00ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.27ms median_wire=12.34ms median_wall=47.29ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.76s | passed=450 failed=0 total=450 avg_case=6.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.06s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.83s | passed=1 failed=0 |
