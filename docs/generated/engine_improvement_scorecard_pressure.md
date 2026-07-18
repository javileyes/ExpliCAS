# Engine Improvement Scorecard

- Generated: 2026-07-17T23:13:17.522511+00:00
- Git branch: main
- Git commit: `682b33a3d9351bc7fcd9a985a64705c169f06634`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.03s avg_case_ms=10.29 simplify=288.53ms avg_simplify_ms=2.89, sum total=200 failed=0 elapsed=902.65ms avg_case_ms=4.51 simplify=295.05ms avg_simplify_ms=1.48, product total=100 failed=0 elapsed=617.86ms avg_case_ms=6.18 simplify=178.74ms avg_simplify_ms=1.79, difference total=50 failed=0 elapsed=404.53ms avg_case_ms=8.09 simplify=124.91ms avg_simplify_ms=2.50
- Engine hotspots: sum simplify=295.05ms avg_simplify_ms=1.48 wall=902.65ms, shifted_quotient simplify=288.53ms avg_simplify_ms=2.89 wall=1.03s, product simplify=178.74ms avg_simplify_ms=1.79 wall=617.86ms, difference simplify=124.91ms avg_simplify_ms=2.50 wall=404.53ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.03s avg_case_ms=10.29 avg_simplify_ms=2.89, sum@0+100 failed=0 elapsed=663.63ms avg_case_ms=6.64 avg_simplify_ms=2.09, product@0+100 failed=0 elapsed=617.86ms avg_case_ms=6.18 avg_simplify_ms=1.79, difference@0+50 failed=0 elapsed=404.53ms avg_case_ms=8.09 avg_simplify_ms=2.50, sum@700+100 failed=0 elapsed=239.02ms avg_case_ms=2.39 avg_simplify_ms=0.86
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.09ms median_wire=17.16ms median_wall=66.60ms, product@0+100 #175 product runs=3 median_simplify=15.35ms median_wire=15.40ms median_wall=59.39ms, sum@0+100 #173 sum runs=3 median_simplify=16.06ms median_wire=16.11ms median_wall=60.39ms, difference@0+50 #174 difference runs=3 median_simplify=15.65ms median_wire=15.70ms median_wall=59.54ms, shifted_quotient@0+100 #112 shifted_quotient runs=3 median_simplify=9.89ms median_wire=9.96ms median_wall=37.45ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.95s | passed=450 failed=0 total=450 avg_case=6.556ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.96s | passed=1 failed=0 |
