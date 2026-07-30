# Engine Improvement Scorecard

- Generated: 2026-07-30T17:34:50.970760+00:00
- Git branch: main
- Git commit: `909e059c2c9229ee336387358485e0059e839c57`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=985.74ms avg_case_ms=9.86 simplify=280.28ms avg_simplify_ms=2.80, sum total=200 failed=0 elapsed=847.09ms avg_case_ms=4.24 simplify=270.99ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=600.71ms avg_case_ms=6.01 simplify=171.90ms avg_simplify_ms=1.72, difference total=50 failed=0 elapsed=392.34ms avg_case_ms=7.85 simplify=114.89ms avg_simplify_ms=2.30
- Engine hotspots: shifted_quotient simplify=280.28ms avg_simplify_ms=2.80 wall=985.74ms, sum simplify=270.99ms avg_simplify_ms=1.35 wall=847.09ms, product simplify=171.90ms avg_simplify_ms=1.72 wall=600.71ms, difference simplify=114.89ms avg_simplify_ms=2.30 wall=392.34ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=985.74ms avg_case_ms=9.86 avg_simplify_ms=2.80, sum@0+100 failed=0 elapsed=617.07ms avg_case_ms=6.17 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=600.71ms avg_case_ms=6.01 avg_simplify_ms=1.72, difference@0+50 failed=0 elapsed=392.34ms avg_case_ms=7.85 avg_simplify_ms=2.30, sum@700+100 failed=0 elapsed=230.02ms avg_case_ms=2.30 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.65ms median_wire=16.72ms median_wall=63.11ms, sum@0+100 #173 sum runs=3 median_simplify=15.06ms median_wire=15.11ms median_wall=57.88ms, product@0+100 #175 product runs=3 median_simplify=15.07ms median_wire=15.12ms median_wall=57.73ms, difference@0+50 #174 difference runs=3 median_simplify=15.06ms median_wire=15.10ms median_wall=57.30ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.83ms median_wire=12.89ms median_wall=48.88ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.83s | passed=450 failed=0 total=450 avg_case=6.289ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.13s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.74s | passed=1 failed=0 |
