# Engine Improvement Scorecard

- Generated: 2026-07-23T13:58:37.682227+00:00
- Git branch: main
- Git commit: `198300bd50c80f830504cd7f14dc1df345aecbda`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.02s avg_case_ms=10.15 simplify=285.78ms avg_simplify_ms=2.86, sum total=200 failed=0 elapsed=919.80ms avg_case_ms=4.60 simplify=310.29ms avg_simplify_ms=1.55, product total=100 failed=0 elapsed=616.99ms avg_case_ms=6.17 simplify=177.64ms avg_simplify_ms=1.78, difference total=50 failed=0 elapsed=410.19ms avg_case_ms=8.20 simplify=124.97ms avg_simplify_ms=2.50
- Engine hotspots: sum simplify=310.29ms avg_simplify_ms=1.55 wall=919.80ms, shifted_quotient simplify=285.78ms avg_simplify_ms=2.86 wall=1.02s, product simplify=177.64ms avg_simplify_ms=1.78 wall=616.99ms, difference simplify=124.97ms avg_simplify_ms=2.50 wall=410.19ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.02s avg_case_ms=10.15 avg_simplify_ms=2.86, sum@0+100 failed=0 elapsed=683.77ms avg_case_ms=6.84 avg_simplify_ms=2.25, product@0+100 failed=0 elapsed=616.99ms avg_case_ms=6.17 avg_simplify_ms=1.78, difference@0+50 failed=0 elapsed=410.19ms avg_case_ms=8.20 avg_simplify_ms=2.50, sum@700+100 failed=0 elapsed=236.03ms avg_case_ms=2.36 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.16ms median_wire=17.25ms median_wall=65.77ms, sum@0+100 #173 sum runs=3 median_simplify=15.46ms median_wire=15.52ms median_wall=59.26ms, difference@0+50 #174 difference runs=3 median_simplify=15.56ms median_wire=15.61ms median_wall=59.17ms, product@0+100 #175 product runs=3 median_simplify=15.70ms median_wire=15.76ms median_wall=59.19ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.25ms median_wire=13.33ms median_wall=50.45ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.96s | passed=450 failed=0 total=450 avg_case=6.578ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.74s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.97s | passed=1 failed=0 |
