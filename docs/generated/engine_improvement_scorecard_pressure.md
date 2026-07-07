# Engine Improvement Scorecard

- Generated: 2026-07-07T19:21:41.477162+00:00
- Git branch: main
- Git commit: `677db98a0f80131d2fa96d954a10276bf9d66992`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=937.67ms avg_case_ms=9.38 simplify=260.43ms avg_simplify_ms=2.60, sum total=200 failed=0 elapsed=827.80ms avg_case_ms=4.14 simplify=267.41ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=578.69ms avg_case_ms=5.79 simplify=164.73ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=382.88ms avg_case_ms=7.66 simplify=115.94ms avg_simplify_ms=2.32
- Engine hotspots: sum simplify=267.41ms avg_simplify_ms=1.34 wall=827.80ms, shifted_quotient simplify=260.43ms avg_simplify_ms=2.60 wall=937.67ms, product simplify=164.73ms avg_simplify_ms=1.65 wall=578.69ms, difference simplify=115.94ms avg_simplify_ms=2.32 wall=382.88ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=937.67ms avg_case_ms=9.38 avg_simplify_ms=2.60, sum@0+100 failed=0 elapsed=601.57ms avg_case_ms=6.02 avg_simplify_ms=1.87, product@0+100 failed=0 elapsed=578.69ms avg_case_ms=5.79 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=382.88ms avg_case_ms=7.66 avg_simplify_ms=2.32, sum@700+100 failed=0 elapsed=226.23ms avg_case_ms=2.26 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.16ms median_wire=16.24ms median_wall=62.36ms, difference@0+50 #174 difference runs=3 median_simplify=14.66ms median_wire=14.71ms median_wall=56.86ms, product@0+100 #175 product runs=3 median_simplify=14.75ms median_wire=14.79ms median_wall=57.37ms, sum@0+100 #173 sum runs=3 median_simplify=14.88ms median_wire=14.92ms median_wall=56.79ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.30ms median_wire=12.37ms median_wall=47.35ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.73s | passed=450 failed=0 total=450 avg_case=6.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.01s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
