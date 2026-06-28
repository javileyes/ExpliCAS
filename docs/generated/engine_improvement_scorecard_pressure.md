# Engine Improvement Scorecard

- Generated: 2026-06-28T10:53:33.953148+00:00
- Git branch: main
- Git commit: `ac115c7128a30648d6133ec4a91fd7eb084ca95f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=803.07ms avg_case_ms=8.03 simplify=231.20ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=712.27ms avg_case_ms=3.56 simplify=245.43ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=481.50ms avg_case_ms=4.81 simplify=141.43ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=333.55ms avg_case_ms=6.67 simplify=109.00ms avg_simplify_ms=2.18
- Engine hotspots: sum simplify=245.43ms avg_simplify_ms=1.23 wall=712.27ms, shifted_quotient simplify=231.20ms avg_simplify_ms=2.31 wall=803.07ms, product simplify=141.43ms avg_simplify_ms=1.41 wall=481.50ms, difference simplify=109.00ms avg_simplify_ms=2.18 wall=333.55ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=803.07ms avg_case_ms=8.03 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=514.31ms avg_case_ms=5.14 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=481.50ms avg_case_ms=4.81 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=333.55ms avg_case_ms=6.67 avg_simplify_ms=2.18, sum@700+100 failed=0 elapsed=197.97ms avg_case_ms=1.98 avg_simplify_ms=0.76
- Steady-state engine reruns: difference@0+50 #174 difference runs=3 median_simplify=11.69ms median_wire=11.74ms median_wall=44.47ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.04ms median_wire=13.11ms median_wall=49.26ms, sum@0+100 #173 sum runs=3 median_simplify=11.68ms median_wire=11.74ms median_wall=44.28ms, product@0+100 #175 product runs=3 median_simplify=11.51ms median_wire=11.56ms median_wall=44.62ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.73ms median_wire=10.80ms median_wall=40.29ms
- Steady-state dominant expressions: difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
