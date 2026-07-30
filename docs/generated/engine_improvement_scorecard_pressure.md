# Engine Improvement Scorecard

- Generated: 2026-07-30T11:48:40.949432+00:00
- Git branch: main
- Git commit: `bb72968e0452c972454cc2f35fd9d05dc00fe378`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=986.90ms avg_case_ms=9.87 simplify=280.09ms avg_simplify_ms=2.80, sum total=200 failed=0 elapsed=876.41ms avg_case_ms=4.38 simplify=283.42ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=612.41ms avg_case_ms=6.12 simplify=177.35ms avg_simplify_ms=1.77, difference total=50 failed=0 elapsed=407.03ms avg_case_ms=8.14 simplify=119.67ms avg_simplify_ms=2.39
- Engine hotspots: sum simplify=283.42ms avg_simplify_ms=1.42 wall=876.41ms, shifted_quotient simplify=280.09ms avg_simplify_ms=2.80 wall=986.90ms, product simplify=177.35ms avg_simplify_ms=1.77 wall=612.41ms, difference simplify=119.67ms avg_simplify_ms=2.39 wall=407.03ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=986.90ms avg_case_ms=9.87 avg_simplify_ms=2.80, sum@0+100 failed=0 elapsed=635.21ms avg_case_ms=6.35 avg_simplify_ms=1.96, product@0+100 failed=0 elapsed=612.41ms avg_case_ms=6.12 avg_simplify_ms=1.77, difference@0+50 failed=0 elapsed=407.03ms avg_case_ms=8.14 avg_simplify_ms=2.39, sum@700+100 failed=0 elapsed=241.20ms avg_case_ms=2.41 avg_simplify_ms=0.87
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.15ms median_wire=17.22ms median_wall=65.62ms, difference@0+50 #174 difference runs=3 median_simplify=15.29ms median_wire=15.34ms median_wall=58.58ms, product@0+100 #175 product runs=3 median_simplify=15.24ms median_wire=15.30ms median_wall=58.21ms, sum@0+100 #173 sum runs=3 median_simplify=15.48ms median_wire=15.54ms median_wall=58.37ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.24ms median_wire=13.31ms median_wall=49.64ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.35s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.76s | passed=1 failed=0 |
