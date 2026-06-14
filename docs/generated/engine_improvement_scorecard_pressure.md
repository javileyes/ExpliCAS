# Engine Improvement Scorecard

- Generated: 2026-06-14T17:30:27.929884+00:00
- Git branch: main
- Git commit: `268ea9df2aae9514de9ff3990373ede800481b0a`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=787.13ms avg_case_ms=7.87 simplify=225.16ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=689.01ms avg_case_ms=3.45 simplify=230.38ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=475.56ms avg_case_ms=4.76 simplify=136.32ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=324.54ms avg_case_ms=6.49 simplify=102.63ms avg_simplify_ms=2.05
- Engine hotspots: sum simplify=230.38ms avg_simplify_ms=1.15 wall=689.01ms, shifted_quotient simplify=225.16ms avg_simplify_ms=2.25 wall=787.13ms, product simplify=136.32ms avg_simplify_ms=1.36 wall=475.56ms, difference simplify=102.63ms avg_simplify_ms=2.05 wall=324.54ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=787.13ms avg_case_ms=7.87 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=500.01ms avg_case_ms=5.00 avg_simplify_ms=1.60, product@0+100 failed=0 elapsed=475.56ms avg_case_ms=4.76 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=324.54ms avg_case_ms=6.49 avg_simplify_ms=2.05, sum@700+100 failed=0 elapsed=189.00ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.87ms median_wire=12.93ms median_wall=49.07ms, difference@0+50 #174 difference runs=3 median_simplify=11.56ms median_wire=11.61ms median_wall=44.32ms, sum@0+100 #173 sum runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.14ms, product@0+100 #175 product runs=3 median_simplify=11.56ms median_wire=11.61ms median_wall=44.34ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.56ms median_wire=10.63ms median_wall=40.31ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
