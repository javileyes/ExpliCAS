# Engine Improvement Scorecard

- Generated: 2026-06-12T13:42:25.158010+00:00
- Git branch: main
- Git commit: `55fc594438bbbc1d98b584d87db7946fcfb87785`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=778.05ms avg_case_ms=7.78 simplify=221.57ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=687.85ms avg_case_ms=3.44 simplify=230.43ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=474.97ms avg_case_ms=4.75 simplify=135.72ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=326.77ms avg_case_ms=6.54 simplify=104.32ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=230.43ms avg_simplify_ms=1.15 wall=687.85ms, shifted_quotient simplify=221.57ms avg_simplify_ms=2.22 wall=778.05ms, product simplify=135.72ms avg_simplify_ms=1.36 wall=474.97ms, difference simplify=104.32ms avg_simplify_ms=2.09 wall=326.77ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=778.05ms avg_case_ms=7.78 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=499.87ms avg_case_ms=5.00 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=474.97ms avg_case_ms=4.75 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=326.77ms avg_case_ms=6.54 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=187.99ms avg_case_ms=1.88 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.93ms median_wire=13.00ms median_wall=49.56ms, product@0+100 #175 product runs=3 median_simplify=11.46ms median_wire=11.51ms median_wall=44.30ms, sum@0+100 #173 sum runs=3 median_simplify=11.71ms median_wire=11.75ms median_wall=44.44ms, difference@0+50 #174 difference runs=3 median_simplify=11.49ms median_wire=11.54ms median_wall=43.81ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.66ms median_wire=10.73ms median_wall=40.28ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.84s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
