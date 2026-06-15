# Engine Improvement Scorecard

- Generated: 2026-06-15T09:15:49.049404+00:00
- Git branch: main
- Git commit: `a05b5bbca2aa3f5b4eb65061472ebf895336a603`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=352

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=787.03ms avg_case_ms=7.87 simplify=224.60ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=694.67ms avg_case_ms=3.47 simplify=235.69ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=493.84ms avg_case_ms=4.94 simplify=145.43ms avg_simplify_ms=1.45, difference total=50 failed=0 elapsed=325.81ms avg_case_ms=6.52 simplify=103.59ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=235.69ms avg_simplify_ms=1.18 wall=694.67ms, shifted_quotient simplify=224.60ms avg_simplify_ms=2.25 wall=787.03ms, product simplify=145.43ms avg_simplify_ms=1.45 wall=493.84ms, difference simplify=103.59ms avg_simplify_ms=2.07 wall=325.81ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=787.03ms avg_case_ms=7.87 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=504.31ms avg_case_ms=5.04 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=493.84ms avg_case_ms=4.94 avg_simplify_ms=1.45, difference@0+50 failed=0 elapsed=325.81ms avg_case_ms=6.52 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=190.36ms avg_case_ms=1.90 avg_simplify_ms=0.71
- Steady-state engine reruns: product@0+100 #175 product runs=3 median_simplify=11.57ms median_wire=11.62ms median_wall=44.41ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.85ms median_wire=12.92ms median_wall=49.25ms, difference@0+50 #174 difference runs=3 median_simplify=11.70ms median_wire=11.76ms median_wall=44.14ms, sum@0+100 #173 sum runs=3 median_simplify=11.59ms median_wire=11.65ms median_wall=43.78ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.73ms median_wire=10.80ms median_wall=40.16ms
- Steady-state dominant expressions: product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.85s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
