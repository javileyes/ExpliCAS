# Engine Improvement Scorecard

- Generated: 2026-06-11T14:49:09.087810+00:00
- Git branch: main
- Git commit: `73213159c6f7068638cfd587530970d15352b708`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=779.78ms avg_case_ms=7.80 simplify=221.60ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=703.28ms avg_case_ms=3.52 simplify=235.03ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=491.27ms avg_case_ms=4.91 simplify=141.73ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=326.47ms avg_case_ms=6.53 simplify=103.52ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=235.03ms avg_simplify_ms=1.18 wall=703.28ms, shifted_quotient simplify=221.60ms avg_simplify_ms=2.22 wall=779.78ms, product simplify=141.73ms avg_simplify_ms=1.42 wall=491.27ms, difference simplify=103.52ms avg_simplify_ms=2.07 wall=326.47ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=779.78ms avg_case_ms=7.80 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=514.83ms avg_case_ms=5.15 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=491.27ms avg_case_ms=4.91 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=326.47ms avg_case_ms=6.53 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=188.45ms avg_case_ms=1.88 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.35ms median_wire=13.43ms median_wall=51.35ms, product@0+100 #175 product runs=3 median_simplify=11.61ms median_wire=11.66ms median_wall=44.77ms, difference@0+50 #174 difference runs=3 median_simplify=11.33ms median_wire=11.38ms median_wall=43.32ms, sum@0+100 #173 sum runs=3 median_simplify=11.54ms median_wire=11.59ms median_wall=44.16ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.29ms median_wire=11.36ms median_wall=41.45ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.84s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.10s | passed=1 failed=0 |
