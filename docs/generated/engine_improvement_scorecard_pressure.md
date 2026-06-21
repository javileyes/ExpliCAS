# Engine Improvement Scorecard

- Generated: 2026-06-21T09:02:27.409698+00:00
- Git branch: main
- Git commit: `173e8b1095a142a4973f46cff8a7d01f168642dc`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=781.89ms avg_case_ms=7.82 simplify=223.62ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=701.92ms avg_case_ms=3.51 simplify=239.84ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=477.63ms avg_case_ms=4.78 simplify=137.56ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=337.31ms avg_case_ms=6.75 simplify=109.12ms avg_simplify_ms=2.18
- Engine hotspots: sum simplify=239.84ms avg_simplify_ms=1.20 wall=701.92ms, shifted_quotient simplify=223.62ms avg_simplify_ms=2.24 wall=781.89ms, product simplify=137.56ms avg_simplify_ms=1.38 wall=477.63ms, difference simplify=109.12ms avg_simplify_ms=2.18 wall=337.31ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=781.89ms avg_case_ms=7.82 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=504.99ms avg_case_ms=5.05 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=477.63ms avg_case_ms=4.78 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=337.31ms avg_case_ms=6.75 avg_simplify_ms=2.18, sum@700+100 failed=0 elapsed=196.93ms avg_case_ms=1.97 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.87ms median_wire=12.93ms median_wall=49.07ms, difference@0+50 #174 difference runs=3 median_simplify=11.87ms median_wire=11.92ms median_wall=44.75ms, product@0+100 #175 product runs=3 median_simplify=11.71ms median_wire=11.76ms median_wall=44.69ms, sum@0+100 #173 sum runs=3 median_simplify=11.60ms median_wire=11.65ms median_wall=43.83ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.53ms median_wire=10.61ms median_wall=40.03ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
