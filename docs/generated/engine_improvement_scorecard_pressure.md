# Engine Improvement Scorecard

- Generated: 2026-06-21T22:43:36.735652+00:00
- Git branch: main
- Git commit: `9ee548d28edf2de730e53672372cd21374c5e4c7`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=795.96ms avg_case_ms=7.96 simplify=227.76ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=703.82ms avg_case_ms=3.52 simplify=239.26ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=483.56ms avg_case_ms=4.84 simplify=139.67ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=331.27ms avg_case_ms=6.63 simplify=105.82ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=239.26ms avg_simplify_ms=1.20 wall=703.82ms, shifted_quotient simplify=227.76ms avg_simplify_ms=2.28 wall=795.96ms, product simplify=139.67ms avg_simplify_ms=1.40 wall=483.56ms, difference simplify=105.82ms avg_simplify_ms=2.12 wall=331.27ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=795.96ms avg_case_ms=7.96 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=508.54ms avg_case_ms=5.09 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=483.56ms avg_case_ms=4.84 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=331.27ms avg_case_ms=6.63 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=195.28ms avg_case_ms=1.95 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.13ms median_wire=13.20ms median_wall=49.67ms, difference@0+50 #174 difference runs=3 median_simplify=11.80ms median_wire=11.86ms median_wall=44.29ms, sum@0+100 #173 sum runs=3 median_simplify=11.68ms median_wire=11.72ms median_wall=44.10ms, product@0+100 #175 product runs=3 median_simplify=11.64ms median_wire=11.69ms median_wall=44.28ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.75ms median_wire=10.82ms median_wall=40.59ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
