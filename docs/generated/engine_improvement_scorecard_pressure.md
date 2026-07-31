# Engine Improvement Scorecard

- Generated: 2026-07-31T06:12:01.492709+00:00
- Git branch: main
- Git commit: `8778f9ba57ed22b0af9f62408422c45bd564ba84`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=967.82ms avg_case_ms=9.68 simplify=273.46ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=849.79ms avg_case_ms=4.25 simplify=273.02ms avg_simplify_ms=1.37, product total=100 failed=0 elapsed=603.82ms avg_case_ms=6.04 simplify=173.50ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=389.93ms avg_case_ms=7.80 simplify=114.94ms avg_simplify_ms=2.30
- Engine hotspots: shifted_quotient simplify=273.46ms avg_simplify_ms=2.73 wall=967.82ms, sum simplify=273.02ms avg_simplify_ms=1.37 wall=849.79ms, product simplify=173.50ms avg_simplify_ms=1.74 wall=603.82ms, difference simplify=114.94ms avg_simplify_ms=2.30 wall=389.93ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=967.82ms avg_case_ms=9.68 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=617.42ms avg_case_ms=6.17 avg_simplify_ms=1.90, product@0+100 failed=0 elapsed=603.82ms avg_case_ms=6.04 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=389.93ms avg_case_ms=7.80 avg_simplify_ms=2.30, sum@700+100 failed=0 elapsed=232.37ms avg_case_ms=2.32 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.73ms median_wire=16.80ms median_wall=64.34ms, sum@0+100 #173 sum runs=3 median_simplify=15.11ms median_wire=15.15ms median_wall=57.23ms, difference@0+50 #174 difference runs=3 median_simplify=15.06ms median_wire=15.11ms median_wall=57.76ms, product@0+100 #175 product runs=3 median_simplify=15.21ms median_wire=15.26ms median_wall=58.63ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.03ms median_wire=13.10ms median_wall=49.01ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.81s | passed=450 failed=0 total=450 avg_case=6.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.18s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.73s | passed=1 failed=0 |
