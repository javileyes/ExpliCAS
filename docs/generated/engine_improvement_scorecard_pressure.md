# Engine Improvement Scorecard

- Generated: 2026-07-17T16:27:00.929274+00:00
- Git branch: main
- Git commit: `54a43802c9c25f00d1639e3362f700af14fa0ea7`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=991.09ms avg_case_ms=9.91 simplify=279.19ms avg_simplify_ms=2.79, sum total=200 failed=0 elapsed=876.88ms avg_case_ms=4.38 simplify=284.51ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=596.83ms avg_case_ms=5.97 simplify=171.85ms avg_simplify_ms=1.72, difference total=50 failed=0 elapsed=399.11ms avg_case_ms=7.98 simplify=121.94ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=284.51ms avg_simplify_ms=1.42 wall=876.88ms, shifted_quotient simplify=279.19ms avg_simplify_ms=2.79 wall=991.09ms, product simplify=171.85ms avg_simplify_ms=1.72 wall=596.83ms, difference simplify=121.94ms avg_simplify_ms=2.44 wall=399.11ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=991.09ms avg_case_ms=9.91 avg_simplify_ms=2.79, sum@0+100 failed=0 elapsed=646.71ms avg_case_ms=6.47 avg_simplify_ms=2.02, product@0+100 failed=0 elapsed=596.83ms avg_case_ms=5.97 avg_simplify_ms=1.72, difference@0+50 failed=0 elapsed=399.11ms avg_case_ms=7.98 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=230.17ms avg_case_ms=2.30 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.08ms median_wire=17.16ms median_wall=63.95ms, difference@0+50 #174 difference runs=3 median_simplify=15.13ms median_wire=15.18ms median_wall=58.33ms, sum@0+100 #173 sum runs=3 median_simplify=15.39ms median_wire=15.44ms median_wall=59.81ms, product@0+100 #175 product runs=3 median_simplify=15.13ms median_wire=15.18ms median_wall=58.58ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.91ms median_wire=12.98ms median_wall=49.90ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
