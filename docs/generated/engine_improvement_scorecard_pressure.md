# Engine Improvement Scorecard

- Generated: 2026-07-18T11:41:42.583023+00:00
- Git branch: main
- Git commit: `c6d7e5c04e7439f107a78a3902f28086ee1e926b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.02s avg_case_ms=10.16 simplify=283.66ms avg_simplify_ms=2.84, sum total=200 failed=0 elapsed=893.97ms avg_case_ms=4.47 simplify=292.44ms avg_simplify_ms=1.46, product total=100 failed=0 elapsed=621.09ms avg_case_ms=6.21 simplify=179.71ms avg_simplify_ms=1.80, difference total=50 failed=0 elapsed=406.65ms avg_case_ms=8.13 simplify=124.79ms avg_simplify_ms=2.50
- Engine hotspots: sum simplify=292.44ms avg_simplify_ms=1.46 wall=893.97ms, shifted_quotient simplify=283.66ms avg_simplify_ms=2.84 wall=1.02s, product simplify=179.71ms avg_simplify_ms=1.80 wall=621.09ms, difference simplify=124.79ms avg_simplify_ms=2.50 wall=406.65ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.02s avg_case_ms=10.16 avg_simplify_ms=2.84, sum@0+100 failed=0 elapsed=656.32ms avg_case_ms=6.56 avg_simplify_ms=2.05, product@0+100 failed=0 elapsed=621.09ms avg_case_ms=6.21 avg_simplify_ms=1.80, difference@0+50 failed=0 elapsed=406.65ms avg_case_ms=8.13 avg_simplify_ms=2.50, sum@700+100 failed=0 elapsed=237.65ms avg_case_ms=2.38 avg_simplify_ms=0.88
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.65ms median_wire=15.71ms median_wall=60.14ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.34ms median_wire=17.41ms median_wall=65.92ms, product@0+100 #175 product runs=3 median_simplify=15.07ms median_wire=15.12ms median_wall=58.42ms, difference@0+50 #174 difference runs=3 median_simplify=15.64ms median_wire=15.70ms median_wall=59.51ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.50ms median_wire=13.58ms median_wall=50.44ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.94s | passed=450 failed=0 total=450 avg_case=6.533ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.84s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.96s | passed=1 failed=0 |
