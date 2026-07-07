# Engine Improvement Scorecard

- Generated: 2026-07-07T08:23:32.597432+00:00
- Git branch: main
- Git commit: `adfd771d29b0f656652da829a3bd6f11c0f11e19`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=934.78ms avg_case_ms=9.35 simplify=258.68ms avg_simplify_ms=2.59, sum total=200 failed=0 elapsed=835.62ms avg_case_ms=4.18 simplify=269.12ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=584.89ms avg_case_ms=5.85 simplify=165.83ms avg_simplify_ms=1.66, difference total=50 failed=0 elapsed=387.93ms avg_case_ms=7.76 simplify=117.39ms avg_simplify_ms=2.35
- Engine hotspots: sum simplify=269.12ms avg_simplify_ms=1.35 wall=835.62ms, shifted_quotient simplify=258.68ms avg_simplify_ms=2.59 wall=934.78ms, product simplify=165.83ms avg_simplify_ms=1.66 wall=584.89ms, difference simplify=117.39ms avg_simplify_ms=2.35 wall=387.93ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=934.78ms avg_case_ms=9.35 avg_simplify_ms=2.59, sum@0+100 failed=0 elapsed=611.71ms avg_case_ms=6.12 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=584.89ms avg_case_ms=5.85 avg_simplify_ms=1.66, difference@0+50 failed=0 elapsed=387.93ms avg_case_ms=7.76 avg_simplify_ms=2.35, sum@700+100 failed=0 elapsed=223.92ms avg_case_ms=2.24 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.14ms median_wire=16.21ms median_wall=62.86ms, sum@0+100 #173 sum runs=3 median_simplify=14.91ms median_wire=14.96ms median_wall=57.35ms, difference@0+50 #174 difference runs=3 median_simplify=14.92ms median_wire=14.97ms median_wall=58.01ms, product@0+100 #175 product runs=3 median_simplify=15.76ms median_wire=15.81ms median_wall=60.47ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.60ms median_wire=12.67ms median_wall=47.85ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.74s | passed=450 failed=0 total=450 avg_case=6.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.00s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
