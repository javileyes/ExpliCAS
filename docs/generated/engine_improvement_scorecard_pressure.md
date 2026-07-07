# Engine Improvement Scorecard

- Generated: 2026-07-07T09:33:31.105055+00:00
- Git branch: main
- Git commit: `4f1bc013ce6aaaa88262a18eecf8d225de1fa13f`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=928.04ms avg_case_ms=9.28 simplify=256.90ms avg_simplify_ms=2.57, sum total=200 failed=0 elapsed=824.25ms avg_case_ms=4.12 simplify=266.13ms avg_simplify_ms=1.33, product total=100 failed=0 elapsed=577.02ms avg_case_ms=5.77 simplify=163.61ms avg_simplify_ms=1.64, difference total=50 failed=0 elapsed=384.67ms avg_case_ms=7.69 simplify=116.15ms avg_simplify_ms=2.32
- Engine hotspots: sum simplify=266.13ms avg_simplify_ms=1.33 wall=824.25ms, shifted_quotient simplify=256.90ms avg_simplify_ms=2.57 wall=928.04ms, product simplify=163.61ms avg_simplify_ms=1.64 wall=577.02ms, difference simplify=116.15ms avg_simplify_ms=2.32 wall=384.67ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=928.04ms avg_case_ms=9.28 avg_simplify_ms=2.57, sum@0+100 failed=0 elapsed=603.51ms avg_case_ms=6.04 avg_simplify_ms=1.87, product@0+100 failed=0 elapsed=577.02ms avg_case_ms=5.77 avg_simplify_ms=1.64, difference@0+50 failed=0 elapsed=384.67ms avg_case_ms=7.69 avg_simplify_ms=2.32, sum@700+100 failed=0 elapsed=220.74ms avg_case_ms=2.21 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.33ms median_wire=16.39ms median_wall=65.17ms, difference@0+50 #174 difference runs=3 median_simplify=15.14ms median_wire=15.19ms median_wall=58.24ms, sum@0+100 #173 sum runs=3 median_simplify=15.72ms median_wire=15.77ms median_wall=59.19ms, product@0+100 #175 product runs=3 median_simplify=14.69ms median_wire=14.74ms median_wall=57.14ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.14ms median_wire=13.21ms median_wall=50.40ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.71s | passed=450 failed=0 total=450 avg_case=6.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.96s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
