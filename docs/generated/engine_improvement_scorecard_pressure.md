# Engine Improvement Scorecard

- Generated: 2026-06-20T07:15:39.392710+00:00
- Git branch: main
- Git commit: `ae26db402f66b5fccd20e4469e1b6036140aff80`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=789.83ms avg_case_ms=7.90 simplify=226.18ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=720.78ms avg_case_ms=3.60 simplify=245.23ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=480.63ms avg_case_ms=4.81 simplify=139.39ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=331.89ms avg_case_ms=6.64 simplify=105.99ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=245.23ms avg_simplify_ms=1.23 wall=720.78ms, shifted_quotient simplify=226.18ms avg_simplify_ms=2.26 wall=789.83ms, product simplify=139.39ms avg_simplify_ms=1.39 wall=480.63ms, difference simplify=105.99ms avg_simplify_ms=2.12 wall=331.89ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=789.83ms avg_case_ms=7.90 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=522.93ms avg_case_ms=5.23 avg_simplify_ms=1.71, product@0+100 failed=0 elapsed=480.63ms avg_case_ms=4.81 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=331.89ms avg_case_ms=6.64 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=197.85ms avg_case_ms=1.98 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.99ms median_wire=13.06ms median_wall=49.42ms, sum@0+100 #173 sum runs=3 median_simplify=11.84ms median_wire=11.89ms median_wall=45.04ms, difference@0+50 #174 difference runs=3 median_simplify=11.92ms median_wire=11.97ms median_wall=45.05ms, product@0+100 #175 product runs=3 median_simplify=11.61ms median_wire=11.66ms median_wall=44.41ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.86ms median_wire=10.93ms median_wall=40.92ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
