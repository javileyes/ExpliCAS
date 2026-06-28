# Engine Improvement Scorecard

- Generated: 2026-06-28T02:13:33.551779+00:00
- Git branch: main
- Git commit: `effd101e212a96bc525abdcde5323e728ed30ffd`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=791.41ms avg_case_ms=7.91 simplify=227.05ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=707.78ms avg_case_ms=3.54 simplify=243.31ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=479.76ms avg_case_ms=4.80 simplify=140.85ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=326.93ms avg_case_ms=6.54 simplify=105.23ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=243.31ms avg_simplify_ms=1.22 wall=707.78ms, shifted_quotient simplify=227.05ms avg_simplify_ms=2.27 wall=791.41ms, product simplify=140.85ms avg_simplify_ms=1.41 wall=479.76ms, difference simplify=105.23ms avg_simplify_ms=2.10 wall=326.93ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=791.41ms avg_case_ms=7.91 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=506.88ms avg_case_ms=5.07 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=479.76ms avg_case_ms=4.80 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=326.93ms avg_case_ms=6.54 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=200.90ms avg_case_ms=2.01 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.04ms median_wire=13.12ms median_wall=49.28ms, product@0+100 #175 product runs=3 median_simplify=11.62ms median_wire=11.68ms median_wall=44.05ms, sum@0+100 #173 sum runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=44.19ms, difference@0+50 #174 difference runs=3 median_simplify=11.56ms median_wire=11.62ms median_wall=44.01ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.44ms median_wire=10.51ms median_wall=39.83ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
