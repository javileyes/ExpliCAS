# Engine Improvement Scorecard

- Generated: 2026-07-31T05:51:56.007322+00:00
- Git branch: main
- Git commit: `1ecaedb8838d63700f6913cdc4e1382e3b969886`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=968.40ms avg_case_ms=9.68 simplify=271.85ms avg_simplify_ms=2.72, sum total=200 failed=0 elapsed=847.17ms avg_case_ms=4.24 simplify=272.56ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=603.82ms avg_case_ms=6.04 simplify=174.14ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=395.33ms avg_case_ms=7.91 simplify=115.62ms avg_simplify_ms=2.31
- Engine hotspots: sum simplify=272.56ms avg_simplify_ms=1.36 wall=847.17ms, shifted_quotient simplify=271.85ms avg_simplify_ms=2.72 wall=968.40ms, product simplify=174.14ms avg_simplify_ms=1.74 wall=603.82ms, difference simplify=115.62ms avg_simplify_ms=2.31 wall=395.33ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=968.40ms avg_case_ms=9.68 avg_simplify_ms=2.72, sum@0+100 failed=0 elapsed=617.54ms avg_case_ms=6.18 avg_simplify_ms=1.90, product@0+100 failed=0 elapsed=603.82ms avg_case_ms=6.04 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=395.33ms avg_case_ms=7.91 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=229.63ms avg_case_ms=2.30 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.79ms median_wire=16.85ms median_wall=64.46ms, sum@0+100 #173 sum runs=3 median_simplify=14.94ms median_wire=14.99ms median_wall=58.12ms, product@0+100 #175 product runs=3 median_simplify=15.25ms median_wire=15.30ms median_wall=57.55ms, difference@0+50 #174 difference runs=3 median_simplify=14.88ms median_wire=14.93ms median_wall=56.94ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.93ms median_wire=13.01ms median_wall=49.28ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.82s | passed=450 failed=0 total=450 avg_case=6.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.24s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.73s | passed=1 failed=0 |
