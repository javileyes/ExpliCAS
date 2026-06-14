# Engine Improvement Scorecard

- Generated: 2026-06-14T09:55:17.151708+00:00
- Git branch: main
- Git commit: `f6678642b0e98d96e0789ad476f80da9bf0a9327`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=796.09ms avg_case_ms=7.96 simplify=225.29ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=686.64ms avg_case_ms=3.43 simplify=231.09ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=472.99ms avg_case_ms=4.73 simplify=135.14ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=325.43ms avg_case_ms=6.51 simplify=103.01ms avg_simplify_ms=2.06
- Engine hotspots: sum simplify=231.09ms avg_simplify_ms=1.16 wall=686.64ms, shifted_quotient simplify=225.29ms avg_simplify_ms=2.25 wall=796.09ms, product simplify=135.14ms avg_simplify_ms=1.35 wall=472.99ms, difference simplify=103.01ms avg_simplify_ms=2.06 wall=325.43ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=796.09ms avg_case_ms=7.96 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=494.67ms avg_case_ms=4.95 avg_simplify_ms=1.59, product@0+100 failed=0 elapsed=472.99ms avg_case_ms=4.73 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=325.43ms avg_case_ms=6.51 avg_simplify_ms=2.06, sum@700+100 failed=0 elapsed=191.97ms avg_case_ms=1.92 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.91ms median_wire=12.99ms median_wall=49.72ms, sum@0+100 #173 sum runs=3 median_simplify=11.48ms median_wire=11.53ms median_wall=43.53ms, difference@0+50 #174 difference runs=3 median_simplify=11.51ms median_wire=11.56ms median_wall=45.27ms, product@0+100 #175 product runs=3 median_simplify=11.88ms median_wire=11.94ms median_wall=44.81ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.68ms median_wire=10.77ms median_wall=40.05ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
