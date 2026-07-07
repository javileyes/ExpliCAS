# Engine Improvement Scorecard

- Generated: 2026-07-07T09:03:30.786672+00:00
- Git branch: main
- Git commit: `eaf3c78aa7fbaf4980bf30342b13954e704b74a5`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=956.52ms avg_case_ms=9.57 simplify=265.61ms avg_simplify_ms=2.66, sum total=200 failed=0 elapsed=869.39ms avg_case_ms=4.35 simplify=279.15ms avg_simplify_ms=1.40, product total=100 failed=0 elapsed=595.11ms avg_case_ms=5.95 simplify=169.01ms avg_simplify_ms=1.69, difference total=50 failed=0 elapsed=399.36ms avg_case_ms=7.99 simplify=120.82ms avg_simplify_ms=2.42
- Engine hotspots: sum simplify=279.15ms avg_simplify_ms=1.40 wall=869.39ms, shifted_quotient simplify=265.61ms avg_simplify_ms=2.66 wall=956.52ms, product simplify=169.01ms avg_simplify_ms=1.69 wall=595.11ms, difference simplify=120.82ms avg_simplify_ms=2.42 wall=399.36ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=956.52ms avg_case_ms=9.57 avg_simplify_ms=2.66, sum@0+100 failed=0 elapsed=629.89ms avg_case_ms=6.30 avg_simplify_ms=1.94, product@0+100 failed=0 elapsed=595.11ms avg_case_ms=5.95 avg_simplify_ms=1.69, difference@0+50 failed=0 elapsed=399.36ms avg_case_ms=7.99 avg_simplify_ms=2.42, sum@700+100 failed=0 elapsed=239.50ms avg_case_ms=2.39 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.21ms median_wire=16.28ms median_wall=62.76ms, sum@0+100 #173 sum runs=3 median_simplify=14.60ms median_wire=14.64ms median_wall=57.91ms, product@0+100 #175 product runs=3 median_simplify=15.50ms median_wire=15.55ms median_wall=59.36ms, difference@0+50 #174 difference runs=3 median_simplify=14.64ms median_wire=14.69ms median_wall=56.07ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.37ms median_wire=12.44ms median_wall=47.33ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.82s | passed=450 failed=0 total=450 avg_case=6.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.00s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
