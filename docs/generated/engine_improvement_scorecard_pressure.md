# Engine Improvement Scorecard

- Generated: 2026-07-28T17:25:38.684686+00:00
- Git branch: main
- Git commit: `c1d823128aee7b86fd9bcfc0f47dc9301f869c16`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=960.09ms avg_case_ms=9.60 simplify=270.98ms avg_simplify_ms=2.71, sum total=200 failed=0 elapsed=864.61ms avg_case_ms=4.32 simplify=281.24ms avg_simplify_ms=1.41, product total=100 failed=0 elapsed=597.93ms avg_case_ms=5.98 simplify=172.24ms avg_simplify_ms=1.72, difference total=50 failed=0 elapsed=399.21ms avg_case_ms=7.98 simplify=121.95ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=281.24ms avg_simplify_ms=1.41 wall=864.61ms, shifted_quotient simplify=270.98ms avg_simplify_ms=2.71 wall=960.09ms, product simplify=172.24ms avg_simplify_ms=1.72 wall=597.93ms, difference simplify=121.95ms avg_simplify_ms=2.44 wall=399.21ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=960.09ms avg_case_ms=9.60 avg_simplify_ms=2.71, sum@0+100 failed=0 elapsed=634.12ms avg_case_ms=6.34 avg_simplify_ms=1.98, product@0+100 failed=0 elapsed=597.93ms avg_case_ms=5.98 avg_simplify_ms=1.72, difference@0+50 failed=0 elapsed=399.21ms avg_case_ms=7.98 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=230.49ms avg_case_ms=2.30 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.55ms median_wire=16.62ms median_wall=64.04ms, sum@0+100 #173 sum runs=3 median_simplify=15.00ms median_wire=15.04ms median_wall=57.48ms, difference@0+50 #174 difference runs=3 median_simplify=14.93ms median_wire=14.98ms median_wall=57.50ms, product@0+100 #175 product runs=3 median_simplify=14.87ms median_wire=14.92ms median_wall=57.08ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.76ms median_wire=12.83ms median_wall=48.31ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.82s | passed=450 failed=0 total=450 avg_case=6.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.20s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
