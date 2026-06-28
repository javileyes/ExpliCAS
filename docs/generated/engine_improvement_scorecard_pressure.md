# Engine Improvement Scorecard

- Generated: 2026-06-28T11:12:45.159501+00:00
- Git branch: main
- Git commit: `375735d2f31aa993452ab37ee151b731db9d5d44`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=838.67ms avg_case_ms=8.39 simplify=239.27ms avg_simplify_ms=2.39, sum total=200 failed=0 elapsed=720.23ms avg_case_ms=3.60 simplify=250.35ms avg_simplify_ms=1.25, product total=100 failed=0 elapsed=496.67ms avg_case_ms=4.97 simplify=146.96ms avg_simplify_ms=1.47, difference total=50 failed=0 elapsed=338.89ms avg_case_ms=6.78 simplify=110.42ms avg_simplify_ms=2.21
- Engine hotspots: sum simplify=250.35ms avg_simplify_ms=1.25 wall=720.23ms, shifted_quotient simplify=239.27ms avg_simplify_ms=2.39 wall=838.67ms, product simplify=146.96ms avg_simplify_ms=1.47 wall=496.67ms, difference simplify=110.42ms avg_simplify_ms=2.21 wall=338.89ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=838.67ms avg_case_ms=8.39 avg_simplify_ms=2.39, sum@0+100 failed=0 elapsed=517.30ms avg_case_ms=5.17 avg_simplify_ms=1.72, product@0+100 failed=0 elapsed=496.67ms avg_case_ms=4.97 avg_simplify_ms=1.47, difference@0+50 failed=0 elapsed=338.89ms avg_case_ms=6.78 avg_simplify_ms=2.21, sum@700+100 failed=0 elapsed=202.94ms avg_case_ms=2.03 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.50ms median_wire=13.58ms median_wall=51.38ms, product@0+100 #175 product runs=3 median_simplify=12.09ms median_wire=12.15ms median_wall=45.92ms, sum@0+100 #173 sum runs=3 median_simplify=12.24ms median_wire=12.29ms median_wall=46.04ms, difference@0+50 #174 difference runs=3 median_simplify=12.13ms median_wire=12.19ms median_wall=45.64ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.04ms median_wire=11.12ms median_wall=41.65ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.40s | passed=450 failed=0 total=450 avg_case=5.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.54s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.10s | passed=1 failed=0 |
