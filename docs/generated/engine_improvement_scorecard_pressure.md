# Engine Improvement Scorecard

- Generated: 2026-07-16T09:46:21.030819+00:00
- Git branch: main
- Git commit: `b6fc54f377322c8b1fe4447af4519766d04920ee`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=366

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.03s avg_case_ms=10.33 simplify=291.56ms avg_simplify_ms=2.92, sum total=200 failed=0 elapsed=911.13ms avg_case_ms=4.56 simplify=294.60ms avg_simplify_ms=1.47, product total=100 failed=0 elapsed=632.94ms avg_case_ms=6.33 simplify=181.93ms avg_simplify_ms=1.82, difference total=50 failed=0 elapsed=416.01ms avg_case_ms=8.32 simplify=127.82ms avg_simplify_ms=2.56
- Engine hotspots: sum simplify=294.60ms avg_simplify_ms=1.47 wall=911.13ms, shifted_quotient simplify=291.56ms avg_simplify_ms=2.92 wall=1.03s, product simplify=181.93ms avg_simplify_ms=1.82 wall=632.94ms, difference simplify=127.82ms avg_simplify_ms=2.56 wall=416.01ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.03s avg_case_ms=10.33 avg_simplify_ms=2.92, sum@0+100 failed=0 elapsed=674.69ms avg_case_ms=6.75 avg_simplify_ms=2.10, product@0+100 failed=0 elapsed=632.94ms avg_case_ms=6.33 avg_simplify_ms=1.82, difference@0+50 failed=0 elapsed=416.01ms avg_case_ms=8.32 avg_simplify_ms=2.56, sum@700+100 failed=0 elapsed=236.44ms avg_case_ms=2.36 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.69ms median_wire=17.76ms median_wall=66.95ms, sum@0+100 #173 sum runs=3 median_simplify=16.01ms median_wire=16.07ms median_wall=59.24ms, difference@0+50 #174 difference runs=3 median_simplify=16.15ms median_wire=16.21ms median_wall=60.22ms, product@0+100 #175 product runs=3 median_simplify=16.16ms median_wire=16.23ms median_wall=60.64ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.51ms median_wire=13.58ms median_wall=50.74ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.99s | passed=450 failed=0 total=450 avg_case=6.644ms |
| `calculus_diff_exhaustive_contract` | `pass` | 13.12s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.96s | passed=1 failed=0 |
