# Engine Improvement Scorecard

- Generated: 2026-06-14T21:41:01.424223+00:00
- Git branch: main
- Git commit: `4ac50b93504fee13b1c87ff3e0731dd7276a52cc`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=351

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=810.41ms avg_case_ms=8.10 simplify=230.36ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=713.09ms avg_case_ms=3.57 simplify=238.77ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=500.58ms avg_case_ms=5.01 simplify=145.19ms avg_simplify_ms=1.45, difference total=50 failed=0 elapsed=335.12ms avg_case_ms=6.70 simplify=106.01ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=238.77ms avg_simplify_ms=1.19 wall=713.09ms, shifted_quotient simplify=230.36ms avg_simplify_ms=2.30 wall=810.41ms, product simplify=145.19ms avg_simplify_ms=1.45 wall=500.58ms, difference simplify=106.01ms avg_simplify_ms=2.12 wall=335.12ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=810.41ms avg_case_ms=8.10 avg_simplify_ms=2.30, sum@0+100 failed=0 elapsed=516.41ms avg_case_ms=5.16 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=500.58ms avg_case_ms=5.01 avg_simplify_ms=1.45, difference@0+50 failed=0 elapsed=335.12ms avg_case_ms=6.70 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=196.67ms avg_case_ms=1.97 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.15ms median_wire=13.23ms median_wall=48.70ms, product@0+100 #175 product runs=3 median_simplify=11.85ms median_wire=11.90ms median_wall=44.90ms, difference@0+50 #174 difference runs=3 median_simplify=11.90ms median_wire=11.96ms median_wall=45.29ms, sum@0+100 #173 sum runs=3 median_simplify=11.84ms median_wire=11.90ms median_wall=45.14ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.72ms median_wire=10.80ms median_wall=40.63ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.36s | passed=450 failed=0 total=450 avg_case=5.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.88s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.97s | passed=1 failed=0 |
