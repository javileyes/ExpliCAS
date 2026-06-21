# Engine Improvement Scorecard

- Generated: 2026-06-21T09:36:23.129929+00:00
- Git branch: main
- Git commit: `be4d5f5747bae737ee9c26c84f99ca59f26a1fe4`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=775.41ms avg_case_ms=7.75 simplify=220.39ms avg_simplify_ms=2.20, sum total=200 failed=0 elapsed=703.04ms avg_case_ms=3.52 simplify=237.65ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=472.61ms avg_case_ms=4.73 simplify=136.53ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=328.14ms avg_case_ms=6.56 simplify=105.02ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=237.65ms avg_simplify_ms=1.19 wall=703.04ms, shifted_quotient simplify=220.39ms avg_simplify_ms=2.20 wall=775.41ms, product simplify=136.53ms avg_simplify_ms=1.37 wall=472.61ms, difference simplify=105.02ms avg_simplify_ms=2.10 wall=328.14ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=775.41ms avg_case_ms=7.75 avg_simplify_ms=2.20, sum@0+100 failed=0 elapsed=515.07ms avg_case_ms=5.15 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=472.61ms avg_case_ms=4.73 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=328.14ms avg_case_ms=6.56 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=187.97ms avg_case_ms=1.88 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.67ms median_wire=12.74ms median_wall=48.90ms, difference@0+50 #174 difference runs=3 median_simplify=11.34ms median_wire=11.39ms median_wall=43.70ms, product@0+100 #175 product runs=3 median_simplify=11.57ms median_wire=11.62ms median_wall=44.88ms, sum@0+100 #173 sum runs=3 median_simplify=11.53ms median_wire=11.58ms median_wall=44.50ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.77ms median_wire=10.85ms median_wall=40.56ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.41s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
