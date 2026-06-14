# Engine Improvement Scorecard

- Generated: 2026-06-14T10:14:11.709386+00:00
- Git branch: main
- Git commit: `d4fc883ac34d4c281605f4a288be57a86913ce46`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=787.91ms avg_case_ms=7.88 simplify=223.35ms avg_simplify_ms=2.23, sum total=200 failed=0 elapsed=686.89ms avg_case_ms=3.43 simplify=231.14ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=472.94ms avg_case_ms=4.73 simplify=135.71ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=323.09ms avg_case_ms=6.46 simplify=102.05ms avg_simplify_ms=2.04
- Engine hotspots: sum simplify=231.14ms avg_simplify_ms=1.16 wall=686.89ms, shifted_quotient simplify=223.35ms avg_simplify_ms=2.23 wall=787.91ms, product simplify=135.71ms avg_simplify_ms=1.36 wall=472.94ms, difference simplify=102.05ms avg_simplify_ms=2.04 wall=323.09ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=787.91ms avg_case_ms=7.88 avg_simplify_ms=2.23, sum@0+100 failed=0 elapsed=495.82ms avg_case_ms=4.96 avg_simplify_ms=1.60, product@0+100 failed=0 elapsed=472.94ms avg_case_ms=4.73 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=323.09ms avg_case_ms=6.46 avg_simplify_ms=2.04, sum@700+100 failed=0 elapsed=191.07ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.94ms median_wire=13.02ms median_wall=49.68ms, difference@0+50 #174 difference runs=3 median_simplify=11.49ms median_wire=11.55ms median_wall=43.35ms, sum@0+100 #173 sum runs=3 median_simplify=11.63ms median_wire=11.67ms median_wall=43.93ms, product@0+100 #175 product runs=3 median_simplify=11.45ms median_wire=11.50ms median_wall=44.10ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.45ms median_wire=10.53ms median_wall=39.80ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
