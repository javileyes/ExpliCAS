# Engine Improvement Scorecard

- Generated: 2026-07-18T15:12:37.170536+00:00
- Git branch: main
- Git commit: `1da047b807d5abde5a5c0934152841eac570eb9f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=971.40ms avg_case_ms=9.71 simplify=271.80ms avg_simplify_ms=2.72, sum total=200 failed=0 elapsed=874.71ms avg_case_ms=4.37 simplify=280.72ms avg_simplify_ms=1.40, product total=100 failed=0 elapsed=602.12ms avg_case_ms=6.02 simplify=172.99ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=396.46ms avg_case_ms=7.93 simplify=121.32ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=280.72ms avg_simplify_ms=1.40 wall=874.71ms, shifted_quotient simplify=271.80ms avg_simplify_ms=2.72 wall=971.40ms, product simplify=172.99ms avg_simplify_ms=1.73 wall=602.12ms, difference simplify=121.32ms avg_simplify_ms=2.43 wall=396.46ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=971.40ms avg_case_ms=9.71 avg_simplify_ms=2.72, sum@0+100 failed=0 elapsed=647.76ms avg_case_ms=6.48 avg_simplify_ms=1.99, product@0+100 failed=0 elapsed=602.12ms avg_case_ms=6.02 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=396.46ms avg_case_ms=7.93 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=226.95ms avg_case_ms=2.27 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.75ms median_wire=16.82ms median_wall=63.75ms, difference@0+50 #174 difference runs=3 median_simplify=15.81ms median_wire=15.87ms median_wall=60.09ms, sum@0+100 #173 sum runs=3 median_simplify=15.42ms median_wire=15.47ms median_wall=58.55ms, product@0+100 #175 product runs=3 median_simplify=15.15ms median_wire=15.21ms median_wall=57.63ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.90ms median_wire=12.97ms median_wall=49.15ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.85s | passed=450 failed=0 total=450 avg_case=6.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
