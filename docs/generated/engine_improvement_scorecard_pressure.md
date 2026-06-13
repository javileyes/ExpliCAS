# Engine Improvement Scorecard

- Generated: 2026-06-13T20:05:02.884725+00:00
- Git branch: main
- Git commit: `8b57e5a4a2ea5dcae80c4a27ca27b1123988c5b2`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=811.53ms avg_case_ms=8.12 simplify=230.08ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=692.44ms avg_case_ms=3.46 simplify=232.49ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=479.41ms avg_case_ms=4.79 simplify=138.49ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=327.33ms avg_case_ms=6.55 simplify=102.74ms avg_simplify_ms=2.05
- Engine hotspots: sum simplify=232.49ms avg_simplify_ms=1.16 wall=692.44ms, shifted_quotient simplify=230.08ms avg_simplify_ms=2.30 wall=811.53ms, product simplify=138.49ms avg_simplify_ms=1.38 wall=479.41ms, difference simplify=102.74ms avg_simplify_ms=2.05 wall=327.33ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=811.53ms avg_case_ms=8.12 avg_simplify_ms=2.30, sum@0+100 failed=0 elapsed=502.29ms avg_case_ms=5.02 avg_simplify_ms=1.62, product@0+100 failed=0 elapsed=479.41ms avg_case_ms=4.79 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=327.33ms avg_case_ms=6.55 avg_simplify_ms=2.05, sum@700+100 failed=0 elapsed=190.15ms avg_case_ms=1.90 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.38ms median_wire=13.46ms median_wall=50.91ms, product@0+100 #175 product runs=3 median_simplify=11.71ms median_wire=11.76ms median_wall=44.39ms, difference@0+50 #174 difference runs=3 median_simplify=11.87ms median_wire=11.92ms median_wall=45.14ms, sum@0+100 #173 sum runs=3 median_simplify=12.12ms median_wire=12.17ms median_wall=45.69ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.29ms median_wire=10.36ms median_wall=39.51ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
