# Engine Improvement Scorecard

- Generated: 2026-08-02T08:41:09.260624+00:00
- Git branch: main
- Git commit: `2745cb10c8dbc1f6e5d2cc0cb8c4395486b2349b`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=391

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=986.96ms avg_case_ms=9.87 simplify=278.62ms avg_simplify_ms=2.79, sum total=200 failed=0 elapsed=859.99ms avg_case_ms=4.30 simplify=278.40ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=612.86ms avg_case_ms=6.13 simplify=177.09ms avg_simplify_ms=1.77, difference total=50 failed=0 elapsed=399.66ms avg_case_ms=7.99 simplify=117.33ms avg_simplify_ms=2.35
- Engine hotspots: shifted_quotient simplify=278.62ms avg_simplify_ms=2.79 wall=986.96ms, sum simplify=278.40ms avg_simplify_ms=1.39 wall=859.99ms, product simplify=177.09ms avg_simplify_ms=1.77 wall=612.86ms, difference simplify=117.33ms avg_simplify_ms=2.35 wall=399.66ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=986.96ms avg_case_ms=9.87 avg_simplify_ms=2.79, sum@0+100 failed=0 elapsed=625.28ms avg_case_ms=6.25 avg_simplify_ms=1.94, product@0+100 failed=0 elapsed=612.86ms avg_case_ms=6.13 avg_simplify_ms=1.77, difference@0+50 failed=0 elapsed=399.66ms avg_case_ms=7.99 avg_simplify_ms=2.35, sum@700+100 failed=0 elapsed=234.71ms avg_case_ms=2.35 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.92ms median_wire=16.99ms median_wall=64.37ms, sum@0+100 #173 sum runs=3 median_simplify=15.05ms median_wire=15.10ms median_wall=57.90ms, product@0+100 #175 product runs=3 median_simplify=15.10ms median_wire=15.16ms median_wall=57.82ms, difference@0+50 #174 difference runs=3 median_simplify=15.21ms median_wire=15.25ms median_wall=57.90ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.17ms median_wire=13.24ms median_wall=49.72ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.22s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.75s | passed=1 failed=0 |
