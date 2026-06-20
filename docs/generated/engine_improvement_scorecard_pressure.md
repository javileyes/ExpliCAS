# Engine Improvement Scorecard

- Generated: 2026-06-20T08:42:01.021750+00:00
- Git branch: main
- Git commit: `a95a3aedc469744ea6977a4019a93e1264405756`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=804.22ms avg_case_ms=8.04 simplify=229.42ms avg_simplify_ms=2.29, sum total=200 failed=0 elapsed=702.87ms avg_case_ms=3.51 simplify=238.61ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=490.65ms avg_case_ms=4.91 simplify=140.09ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=336.68ms avg_case_ms=6.73 simplify=107.33ms avg_simplify_ms=2.15
- Engine hotspots: sum simplify=238.61ms avg_simplify_ms=1.19 wall=702.87ms, shifted_quotient simplify=229.42ms avg_simplify_ms=2.29 wall=804.22ms, product simplify=140.09ms avg_simplify_ms=1.40 wall=490.65ms, difference simplify=107.33ms avg_simplify_ms=2.15 wall=336.68ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=804.22ms avg_case_ms=8.04 avg_simplify_ms=2.29, sum@0+100 failed=0 elapsed=509.86ms avg_case_ms=5.10 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=490.65ms avg_case_ms=4.91 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=336.68ms avg_case_ms=6.73 avg_simplify_ms=2.15, sum@700+100 failed=0 elapsed=193.01ms avg_case_ms=1.93 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.87ms median_wire=13.94ms median_wall=52.99ms, difference@0+50 #174 difference runs=3 median_simplify=12.11ms median_wire=12.17ms median_wall=45.42ms, sum@0+100 #173 sum runs=3 median_simplify=12.03ms median_wire=12.08ms median_wall=45.78ms, product@0+100 #175 product runs=3 median_simplify=11.82ms median_wire=11.88ms median_wall=45.07ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.92ms median_wire=11.01ms median_wall=41.07ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
