# Engine Improvement Scorecard

- Generated: 2026-07-30T17:17:52.650110+00:00
- Git branch: main
- Git commit: `6ee9772a79289a57e2561b45d880335e7622928b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=962.32ms avg_case_ms=9.62 simplify=271.00ms avg_simplify_ms=2.71, sum total=200 failed=0 elapsed=878.01ms avg_case_ms=4.39 simplify=280.75ms avg_simplify_ms=1.40, product total=100 failed=0 elapsed=596.15ms avg_case_ms=5.96 simplify=171.05ms avg_simplify_ms=1.71, difference total=50 failed=0 elapsed=384.72ms avg_case_ms=7.69 simplify=112.98ms avg_simplify_ms=2.26
- Engine hotspots: sum simplify=280.75ms avg_simplify_ms=1.40 wall=878.01ms, shifted_quotient simplify=271.00ms avg_simplify_ms=2.71 wall=962.32ms, product simplify=171.05ms avg_simplify_ms=1.71 wall=596.15ms, difference simplify=112.98ms avg_simplify_ms=2.26 wall=384.72ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=962.32ms avg_case_ms=9.62 avg_simplify_ms=2.71, sum@0+100 failed=0 elapsed=641.79ms avg_case_ms=6.42 avg_simplify_ms=1.97, product@0+100 failed=0 elapsed=596.15ms avg_case_ms=5.96 avg_simplify_ms=1.71, difference@0+50 failed=0 elapsed=384.72ms avg_case_ms=7.69 avg_simplify_ms=2.26, sum@700+100 failed=0 elapsed=236.22ms avg_case_ms=2.36 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.60ms median_wire=16.66ms median_wall=63.90ms, sum@0+100 #173 sum runs=3 median_simplify=15.06ms median_wire=15.11ms median_wall=57.17ms, difference@0+50 #174 difference runs=3 median_simplify=15.25ms median_wire=15.30ms median_wall=60.79ms, product@0+100 #175 product runs=3 median_simplify=15.44ms median_wire=15.49ms median_wall=58.78ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.03ms median_wire=13.10ms median_wall=45.96ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.82s | passed=450 failed=0 total=450 avg_case=6.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.13s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.74s | passed=1 failed=0 |
