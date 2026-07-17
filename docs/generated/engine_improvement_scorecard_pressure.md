# Engine Improvement Scorecard

- Generated: 2026-07-17T17:26:35.140048+00:00
- Git branch: main
- Git commit: `df48800bb4be1d934a2af831ea0f25a92d5f8854`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=998.48ms avg_case_ms=9.98 simplify=276.86ms avg_simplify_ms=2.77, sum total=200 failed=0 elapsed=885.32ms avg_case_ms=4.43 simplify=285.78ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=606.79ms avg_case_ms=6.07 simplify=174.24ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=401.69ms avg_case_ms=8.03 simplify=122.07ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=285.78ms avg_simplify_ms=1.43 wall=885.32ms, shifted_quotient simplify=276.86ms avg_simplify_ms=2.77 wall=998.48ms, product simplify=174.24ms avg_simplify_ms=1.74 wall=606.79ms, difference simplify=122.07ms avg_simplify_ms=2.44 wall=401.69ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=998.48ms avg_case_ms=9.98 avg_simplify_ms=2.77, sum@0+100 failed=0 elapsed=648.34ms avg_case_ms=6.48 avg_simplify_ms=2.00, product@0+100 failed=0 elapsed=606.79ms avg_case_ms=6.07 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=401.69ms avg_case_ms=8.03 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=236.98ms avg_case_ms=2.37 avg_simplify_ms=0.86
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.75ms median_wire=16.83ms median_wall=64.00ms, sum@0+100 #173 sum runs=3 median_simplify=15.22ms median_wire=15.27ms median_wall=58.59ms, product@0+100 #175 product runs=3 median_simplify=15.14ms median_wire=15.19ms median_wall=58.39ms, difference@0+50 #174 difference runs=3 median_simplify=15.66ms median_wire=15.72ms median_wall=59.21ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.09ms median_wire=13.17ms median_wall=50.25ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.89s | passed=450 failed=0 total=450 avg_case=6.422ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.53s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
