# Engine Improvement Scorecard

- Generated: 2026-07-08T19:39:01.949385+00:00
- Git branch: main
- Git commit: `8f372dc4624b84ef22be1c771c50adb57d2d04a2`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=933.79ms avg_case_ms=9.34 simplify=260.05ms avg_simplify_ms=2.60, sum total=200 failed=0 elapsed=820.08ms avg_case_ms=4.10 simplify=264.41ms avg_simplify_ms=1.32, product total=100 failed=0 elapsed=576.36ms avg_case_ms=5.76 simplify=165.30ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=384.62ms avg_case_ms=7.69 simplify=116.87ms avg_simplify_ms=2.34
- Engine hotspots: sum simplify=264.41ms avg_simplify_ms=1.32 wall=820.08ms, shifted_quotient simplify=260.05ms avg_simplify_ms=2.60 wall=933.79ms, product simplify=165.30ms avg_simplify_ms=1.65 wall=576.36ms, difference simplify=116.87ms avg_simplify_ms=2.34 wall=384.62ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=933.79ms avg_case_ms=9.34 avg_simplify_ms=2.60, sum@0+100 failed=0 elapsed=600.52ms avg_case_ms=6.01 avg_simplify_ms=1.86, product@0+100 failed=0 elapsed=576.36ms avg_case_ms=5.76 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=384.62ms avg_case_ms=7.69 avg_simplify_ms=2.34, sum@700+100 failed=0 elapsed=219.56ms avg_case_ms=2.20 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.25ms median_wire=16.32ms median_wall=63.55ms, difference@0+50 #174 difference runs=3 median_simplify=14.73ms median_wire=14.78ms median_wall=56.72ms, product@0+100 #175 product runs=3 median_simplify=14.71ms median_wire=14.76ms median_wall=56.23ms, sum@0+100 #173 sum runs=3 median_simplify=14.87ms median_wire=14.91ms median_wall=56.66ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.12ms median_wire=12.19ms median_wall=46.67ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.72s | passed=450 failed=0 total=450 avg_case=6.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.02s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
