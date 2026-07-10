# Engine Improvement Scorecard

- Generated: 2026-07-10T21:10:34.615197+00:00
- Git branch: main
- Git commit: `a7f5c21e48d9093f9a363973e22008779368d3e3`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=946.03ms avg_case_ms=9.46 simplify=263.34ms avg_simplify_ms=2.63, sum total=200 failed=0 elapsed=854.50ms avg_case_ms=4.27 simplify=278.49ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=593.64ms avg_case_ms=5.94 simplify=168.25ms avg_simplify_ms=1.68, difference total=50 failed=0 elapsed=390.55ms avg_case_ms=7.81 simplify=118.88ms avg_simplify_ms=2.38
- Engine hotspots: sum simplify=278.49ms avg_simplify_ms=1.39 wall=854.50ms, shifted_quotient simplify=263.34ms avg_simplify_ms=2.63 wall=946.03ms, product simplify=168.25ms avg_simplify_ms=1.68 wall=593.64ms, difference simplify=118.88ms avg_simplify_ms=2.38 wall=390.55ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=946.03ms avg_case_ms=9.46 avg_simplify_ms=2.63, sum@0+100 failed=0 elapsed=628.56ms avg_case_ms=6.29 avg_simplify_ms=1.97, product@0+100 failed=0 elapsed=593.64ms avg_case_ms=5.94 avg_simplify_ms=1.68, difference@0+50 failed=0 elapsed=390.55ms avg_case_ms=7.81 avg_simplify_ms=2.38, sum@700+100 failed=0 elapsed=225.94ms avg_case_ms=2.26 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.34ms median_wire=16.42ms median_wall=63.44ms, sum@0+100 #173 sum runs=3 median_simplify=14.91ms median_wire=14.96ms median_wall=56.78ms, product@0+100 #175 product runs=3 median_simplify=14.83ms median_wire=14.88ms median_wall=57.87ms, difference@0+50 #174 difference runs=3 median_simplify=15.13ms median_wire=15.17ms median_wall=58.01ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.51ms median_wire=12.58ms median_wall=47.77ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.79s | passed=450 failed=0 total=450 avg_case=6.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.99s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
