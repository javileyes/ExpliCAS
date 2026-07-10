# Engine Improvement Scorecard

- Generated: 2026-07-10T22:33:00.172685+00:00
- Git branch: main
- Git commit: `ac106fad903c1b542024ce77276801136309d70f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=948.54ms avg_case_ms=9.49 simplify=263.72ms avg_simplify_ms=2.64, sum total=200 failed=0 elapsed=837.22ms avg_case_ms=4.19 simplify=270.99ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=587.70ms avg_case_ms=5.88 simplify=167.11ms avg_simplify_ms=1.67, difference total=50 failed=0 elapsed=389.67ms avg_case_ms=7.79 simplify=117.66ms avg_simplify_ms=2.35
- Engine hotspots: sum simplify=270.99ms avg_simplify_ms=1.35 wall=837.22ms, shifted_quotient simplify=263.72ms avg_simplify_ms=2.64 wall=948.54ms, product simplify=167.11ms avg_simplify_ms=1.67 wall=587.70ms, difference simplify=117.66ms avg_simplify_ms=2.35 wall=389.67ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=948.54ms avg_case_ms=9.49 avg_simplify_ms=2.64, sum@0+100 failed=0 elapsed=612.49ms avg_case_ms=6.12 avg_simplify_ms=1.90, product@0+100 failed=0 elapsed=587.70ms avg_case_ms=5.88 avg_simplify_ms=1.67, difference@0+50 failed=0 elapsed=389.67ms avg_case_ms=7.79 avg_simplify_ms=2.35, sum@700+100 failed=0 elapsed=224.72ms avg_case_ms=2.25 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.54ms median_wire=16.61ms median_wall=63.56ms, sum@0+100 #173 sum runs=3 median_simplify=15.17ms median_wire=15.22ms median_wall=57.64ms, product@0+100 #175 product runs=3 median_simplify=15.33ms median_wire=15.39ms median_wall=58.75ms, difference@0+50 #174 difference runs=3 median_simplify=15.12ms median_wire=15.17ms median_wall=57.52ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.60ms median_wire=12.67ms median_wall=48.23ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.76s | passed=450 failed=0 total=450 avg_case=6.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.02s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
