# Engine Improvement Scorecard

- Generated: 2026-06-28T18:32:30.958248+00:00
- Git branch: main
- Git commit: `5e7f2dc2ce2e151246067964c258ab80abaddc58`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=810.56ms avg_case_ms=8.11 simplify=233.68ms avg_simplify_ms=2.34, sum total=200 failed=0 elapsed=735.25ms avg_case_ms=3.68 simplify=256.76ms avg_simplify_ms=1.28, product total=100 failed=0 elapsed=494.57ms avg_case_ms=4.95 simplify=146.34ms avg_simplify_ms=1.46, difference total=50 failed=0 elapsed=338.54ms avg_case_ms=6.77 simplify=110.56ms avg_simplify_ms=2.21
- Engine hotspots: sum simplify=256.76ms avg_simplify_ms=1.28 wall=735.25ms, shifted_quotient simplify=233.68ms avg_simplify_ms=2.34 wall=810.56ms, product simplify=146.34ms avg_simplify_ms=1.46 wall=494.57ms, difference simplify=110.56ms avg_simplify_ms=2.21 wall=338.54ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=810.56ms avg_case_ms=8.11 avg_simplify_ms=2.34, sum@0+100 failed=0 elapsed=532.13ms avg_case_ms=5.32 avg_simplify_ms=1.77, product@0+100 failed=0 elapsed=494.57ms avg_case_ms=4.95 avg_simplify_ms=1.46, difference@0+50 failed=0 elapsed=338.54ms avg_case_ms=6.77 avg_simplify_ms=2.21, sum@700+100 failed=0 elapsed=203.12ms avg_case_ms=2.03 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.37ms median_wire=13.45ms median_wall=51.11ms, product@0+100 #175 product runs=3 median_simplify=12.05ms median_wire=12.11ms median_wall=45.53ms, difference@0+50 #174 difference runs=3 median_simplify=11.80ms median_wire=11.85ms median_wall=45.37ms, sum@0+100 #173 sum runs=3 median_simplify=11.87ms median_wire=11.92ms median_wall=45.03ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.86ms median_wire=10.94ms median_wall=41.40ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.38s | passed=450 failed=0 total=450 avg_case=5.289ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.51s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.10s | passed=1 failed=0 |
