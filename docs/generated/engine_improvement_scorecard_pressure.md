# Engine Improvement Scorecard

- Generated: 2026-06-28T15:15:22.044915+00:00
- Git branch: main
- Git commit: `668e08ee37b8cc441a7c7ba4d2f87585b81fc90c`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=811.87ms avg_case_ms=8.12 simplify=233.98ms avg_simplify_ms=2.34, sum total=200 failed=0 elapsed=725.50ms avg_case_ms=3.63 simplify=251.94ms avg_simplify_ms=1.26, product total=100 failed=0 elapsed=491.14ms avg_case_ms=4.91 simplify=145.22ms avg_simplify_ms=1.45, difference total=50 failed=0 elapsed=339.62ms avg_case_ms=6.79 simplify=110.16ms avg_simplify_ms=2.20
- Engine hotspots: sum simplify=251.94ms avg_simplify_ms=1.26 wall=725.50ms, shifted_quotient simplify=233.98ms avg_simplify_ms=2.34 wall=811.87ms, product simplify=145.22ms avg_simplify_ms=1.45 wall=491.14ms, difference simplify=110.16ms avg_simplify_ms=2.20 wall=339.62ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=811.87ms avg_case_ms=8.12 avg_simplify_ms=2.34, sum@0+100 failed=0 elapsed=521.12ms avg_case_ms=5.21 avg_simplify_ms=1.73, product@0+100 failed=0 elapsed=491.14ms avg_case_ms=4.91 avg_simplify_ms=1.45, difference@0+50 failed=0 elapsed=339.62ms avg_case_ms=6.79 avg_simplify_ms=2.20, sum@700+100 failed=0 elapsed=204.38ms avg_case_ms=2.04 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.24ms median_wire=13.31ms median_wall=50.86ms, sum@0+100 #173 sum runs=3 median_simplify=12.15ms median_wire=12.21ms median_wall=45.50ms, product@0+100 #175 product runs=3 median_simplify=11.68ms median_wire=11.74ms median_wall=44.77ms, difference@0+50 #174 difference runs=3 median_simplify=11.95ms median_wire=12.00ms median_wall=44.99ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.77ms median_wire=10.85ms median_wall=40.66ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.37s | passed=450 failed=0 total=450 avg_case=5.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
