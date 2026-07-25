# Engine Improvement Scorecard

- Generated: 2026-07-25T04:43:31.829940+00:00
- Git branch: main
- Git commit: `04e7cda0f470833fa13fe77ec681f9a351ee3905`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=998.08ms avg_case_ms=9.98 simplify=280.86ms avg_simplify_ms=2.81, sum total=200 failed=0 elapsed=884.23ms avg_case_ms=4.42 simplify=291.17ms avg_simplify_ms=1.46, product total=100 failed=0 elapsed=616.65ms avg_case_ms=6.17 simplify=177.33ms avg_simplify_ms=1.77, difference total=50 failed=0 elapsed=415.48ms avg_case_ms=8.31 simplify=126.65ms avg_simplify_ms=2.53
- Engine hotspots: sum simplify=291.17ms avg_simplify_ms=1.46 wall=884.23ms, shifted_quotient simplify=280.86ms avg_simplify_ms=2.81 wall=998.08ms, product simplify=177.33ms avg_simplify_ms=1.77 wall=616.65ms, difference simplify=126.65ms avg_simplify_ms=2.53 wall=415.48ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=998.08ms avg_case_ms=9.98 avg_simplify_ms=2.81, sum@0+100 failed=0 elapsed=644.35ms avg_case_ms=6.44 avg_simplify_ms=2.05, product@0+100 failed=0 elapsed=616.65ms avg_case_ms=6.17 avg_simplify_ms=1.77, difference@0+50 failed=0 elapsed=415.48ms avg_case_ms=8.31 avg_simplify_ms=2.53, sum@700+100 failed=0 elapsed=239.88ms avg_case_ms=2.40 avg_simplify_ms=0.86
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.29ms median_wire=17.36ms median_wall=66.60ms, sum@0+100 #173 sum runs=3 median_simplify=15.62ms median_wire=15.67ms median_wall=58.68ms, product@0+100 #175 product runs=3 median_simplify=14.99ms median_wire=15.04ms median_wall=59.13ms, difference@0+50 #174 difference runs=3 median_simplify=15.50ms median_wire=15.55ms median_wall=59.93ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.33ms median_wire=13.41ms median_wall=50.52ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.92s | passed=450 failed=0 total=450 avg_case=6.489ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
