# Engine Improvement Scorecard

- Generated: 2026-06-11T17:56:08.979695+00:00
- Git branch: main
- Git commit: `40ce1d2e8fd2cc2d35ce9eb06d3a6bfaa4e39ca8`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=787.01ms avg_case_ms=7.87 simplify=221.83ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=705.42ms avg_case_ms=3.53 simplify=235.73ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=469.12ms avg_case_ms=4.69 simplify=134.89ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=324.43ms avg_case_ms=6.49 simplify=102.69ms avg_simplify_ms=2.05
- Engine hotspots: sum simplify=235.73ms avg_simplify_ms=1.18 wall=705.42ms, shifted_quotient simplify=221.83ms avg_simplify_ms=2.22 wall=787.01ms, product simplify=134.89ms avg_simplify_ms=1.35 wall=469.12ms, difference simplify=102.69ms avg_simplify_ms=2.05 wall=324.43ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=787.01ms avg_case_ms=7.87 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=517.10ms avg_case_ms=5.17 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=469.12ms avg_case_ms=4.69 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=324.43ms avg_case_ms=6.49 avg_simplify_ms=2.05, sum@700+100 failed=0 elapsed=188.32ms avg_case_ms=1.88 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.43ms median_wire=13.50ms median_wall=55.15ms, difference@0+50 #174 difference runs=3 median_simplify=11.67ms median_wire=11.73ms median_wall=44.48ms, sum@0+100 #173 sum runs=3 median_simplify=11.57ms median_wire=11.61ms median_wall=44.03ms, product@0+100 #175 product runs=3 median_simplify=11.56ms median_wire=11.62ms median_wall=46.01ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.70ms median_wire=12.78ms median_wall=45.07ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.84s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.06s | passed=1 failed=0 |
