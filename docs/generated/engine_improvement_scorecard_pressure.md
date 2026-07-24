# Engine Improvement Scorecard

- Generated: 2026-07-24T15:56:26.685561+00:00
- Git branch: main
- Git commit: `d9da1c8a5e585c998a3eb3cba2830ced78645060`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.01s avg_case_ms=10.13 simplify=285.86ms avg_simplify_ms=2.86, sum total=200 failed=0 elapsed=906.81ms avg_case_ms=4.53 simplify=296.05ms avg_simplify_ms=1.48, product total=100 failed=0 elapsed=608.50ms avg_case_ms=6.09 simplify=174.89ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=407.56ms avg_case_ms=8.15 simplify=124.32ms avg_simplify_ms=2.49
- Engine hotspots: sum simplify=296.05ms avg_simplify_ms=1.48 wall=906.81ms, shifted_quotient simplify=285.86ms avg_simplify_ms=2.86 wall=1.01s, product simplify=174.89ms avg_simplify_ms=1.75 wall=608.50ms, difference simplify=124.32ms avg_simplify_ms=2.49 wall=407.56ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.01s avg_case_ms=10.13 avg_simplify_ms=2.86, sum@0+100 failed=0 elapsed=670.86ms avg_case_ms=6.71 avg_simplify_ms=2.11, product@0+100 failed=0 elapsed=608.50ms avg_case_ms=6.09 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=407.56ms avg_case_ms=8.15 avg_simplify_ms=2.49, sum@700+100 failed=0 elapsed=235.95ms avg_case_ms=2.36 avg_simplify_ms=0.85
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.40ms median_wire=15.44ms median_wall=58.88ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.88ms median_wire=16.96ms median_wall=65.67ms, sum@0+100 #157 sum runs=3 median_simplify=9.67ms median_wire=9.72ms median_wall=36.44ms, difference@0+50 #174 difference runs=3 median_simplify=15.37ms median_wire=15.41ms median_wall=63.89ms, product@0+100 #175 product runs=3 median_simplify=15.18ms median_wire=15.24ms median_wall=58.61ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #157 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^3) + ln(y^2) - ln(x^3 * y^2))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.94s | passed=450 failed=0 total=450 avg_case=6.533ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
