# Engine Improvement Scorecard

- Generated: 2026-06-15T20:13:19.250907+00:00
- Git branch: main
- Git commit: `39b6591ea3733e2ba2604922101f1cc97fab3a43`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=785.21ms avg_case_ms=7.85 simplify=224.04ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=702.37ms avg_case_ms=3.51 simplify=235.95ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=483.09ms avg_case_ms=4.83 simplify=139.09ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=328.39ms avg_case_ms=6.57 simplify=104.44ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=235.95ms avg_simplify_ms=1.18 wall=702.37ms, shifted_quotient simplify=224.04ms avg_simplify_ms=2.24 wall=785.21ms, product simplify=139.09ms avg_simplify_ms=1.39 wall=483.09ms, difference simplify=104.44ms avg_simplify_ms=2.09 wall=328.39ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=785.21ms avg_case_ms=7.85 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=508.46ms avg_case_ms=5.08 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=483.09ms avg_case_ms=4.83 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=328.39ms avg_case_ms=6.57 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=193.91ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.14ms median_wire=13.22ms median_wall=49.83ms, difference@0+50 #174 difference runs=3 median_simplify=11.80ms median_wire=11.85ms median_wall=44.67ms, sum@0+100 #173 sum runs=3 median_simplify=11.48ms median_wire=11.53ms median_wall=44.22ms, product@0+100 #175 product runs=3 median_simplify=11.48ms median_wire=11.54ms median_wall=44.41ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.81ms median_wire=10.89ms median_wall=40.62ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
