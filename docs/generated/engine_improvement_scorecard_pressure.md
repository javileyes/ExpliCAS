# Engine Improvement Scorecard

- Generated: 2026-07-15T20:57:00.665221+00:00
- Git branch: main
- Git commit: `0d2e0f35f8ac137b705ae5e4e57da0ec68ebc3e1`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=364

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=951.18ms avg_case_ms=9.51 simplify=264.63ms avg_simplify_ms=2.65, sum total=200 failed=0 elapsed=861.70ms avg_case_ms=4.31 simplify=279.39ms avg_simplify_ms=1.40, product total=100 failed=0 elapsed=593.48ms avg_case_ms=5.93 simplify=168.89ms avg_simplify_ms=1.69, difference total=50 failed=0 elapsed=395.44ms avg_case_ms=7.91 simplify=119.49ms avg_simplify_ms=2.39
- Engine hotspots: sum simplify=279.39ms avg_simplify_ms=1.40 wall=861.70ms, shifted_quotient simplify=264.63ms avg_simplify_ms=2.65 wall=951.18ms, product simplify=168.89ms avg_simplify_ms=1.69 wall=593.48ms, difference simplify=119.49ms avg_simplify_ms=2.39 wall=395.44ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=951.18ms avg_case_ms=9.51 avg_simplify_ms=2.65, sum@0+100 failed=0 elapsed=632.93ms avg_case_ms=6.33 avg_simplify_ms=1.98, product@0+100 failed=0 elapsed=593.48ms avg_case_ms=5.93 avg_simplify_ms=1.69, difference@0+50 failed=0 elapsed=395.44ms avg_case_ms=7.91 avg_simplify_ms=2.39, sum@700+100 failed=0 elapsed=228.78ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.31ms median_wire=15.36ms median_wall=58.11ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.75ms median_wire=17.82ms median_wall=64.00ms, difference@0+50 #174 difference runs=3 median_simplify=14.92ms median_wire=14.96ms median_wall=57.30ms, product@0+100 #175 product runs=3 median_simplify=15.36ms median_wire=15.41ms median_wall=58.33ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.83ms median_wire=12.90ms median_wall=49.50ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.80s | passed=450 failed=0 total=450 avg_case=6.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.40s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
