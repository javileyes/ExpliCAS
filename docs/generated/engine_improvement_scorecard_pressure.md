# Engine Improvement Scorecard

- Generated: 2026-06-15T06:24:53.869301+00:00
- Git branch: main
- Git commit: `8cf6df7701c536cfcfec55aebb0b21d79431e881`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=791.68ms avg_case_ms=7.92 simplify=225.68ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=696.25ms avg_case_ms=3.48 simplify=233.48ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=499.73ms avg_case_ms=5.00 simplify=143.53ms avg_simplify_ms=1.44, difference total=50 failed=0 elapsed=330.99ms avg_case_ms=6.62 simplify=105.19ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=233.48ms avg_simplify_ms=1.17 wall=696.25ms, shifted_quotient simplify=225.68ms avg_simplify_ms=2.26 wall=791.68ms, product simplify=143.53ms avg_simplify_ms=1.44 wall=499.73ms, difference simplify=105.19ms avg_simplify_ms=2.10 wall=330.99ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=791.68ms avg_case_ms=7.92 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=504.20ms avg_case_ms=5.04 avg_simplify_ms=1.62, product@0+100 failed=0 elapsed=499.73ms avg_case_ms=5.00 avg_simplify_ms=1.44, difference@0+50 failed=0 elapsed=330.99ms avg_case_ms=6.62 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=192.05ms avg_case_ms=1.92 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.91ms median_wire=12.98ms median_wall=49.25ms, product@0+100 #175 product runs=3 median_simplify=11.59ms median_wire=11.64ms median_wall=44.30ms, difference@0+50 #174 difference runs=3 median_simplify=11.76ms median_wire=11.80ms median_wall=44.30ms, sum@0+100 #173 sum runs=3 median_simplify=11.89ms median_wire=11.95ms median_wall=44.72ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.75ms median_wire=10.82ms median_wall=40.70ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.86s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
