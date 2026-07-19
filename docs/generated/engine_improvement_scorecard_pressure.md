# Engine Improvement Scorecard

- Generated: 2026-07-19T12:07:21.043493+00:00
- Git branch: main
- Git commit: `f9f65ab820fc8127762ac255c6e492bf41034416`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.03s avg_case_ms=10.29 simplify=295.62ms avg_simplify_ms=2.96, sum total=200 failed=0 elapsed=904.56ms avg_case_ms=4.52 simplify=294.94ms avg_simplify_ms=1.47, product total=100 failed=0 elapsed=630.26ms avg_case_ms=6.30 simplify=181.81ms avg_simplify_ms=1.82, difference total=50 failed=0 elapsed=414.07ms avg_case_ms=8.28 simplify=127.07ms avg_simplify_ms=2.54
- Engine hotspots: shifted_quotient simplify=295.62ms avg_simplify_ms=2.96 wall=1.03s, sum simplify=294.94ms avg_simplify_ms=1.47 wall=904.56ms, product simplify=181.81ms avg_simplify_ms=1.82 wall=630.26ms, difference simplify=127.07ms avg_simplify_ms=2.54 wall=414.07ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.03s avg_case_ms=10.29 avg_simplify_ms=2.96, sum@0+100 failed=0 elapsed=667.22ms avg_case_ms=6.67 avg_simplify_ms=2.09, product@0+100 failed=0 elapsed=630.26ms avg_case_ms=6.30 avg_simplify_ms=1.82, difference@0+50 failed=0 elapsed=414.07ms avg_case_ms=8.28 avg_simplify_ms=2.54, sum@700+100 failed=0 elapsed=237.34ms avg_case_ms=2.37 avg_simplify_ms=0.86
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.79ms median_wire=15.84ms median_wall=60.00ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.34ms median_wire=17.42ms median_wall=66.10ms, product@0+100 #175 product runs=3 median_simplify=15.76ms median_wire=15.81ms median_wall=60.36ms, difference@0+50 #174 difference runs=3 median_simplify=15.37ms median_wire=15.42ms median_wall=58.72ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.96ms median_wire=13.03ms median_wall=49.91ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.98s | passed=450 failed=0 total=450 avg_case=6.622ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.99s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.96s | passed=1 failed=0 |
