# Engine Improvement Scorecard

- Generated: 2026-07-30T14:41:10.541964+00:00
- Git branch: main
- Git commit: `df10db7b7044357c7703d2e8b207e4970699d521`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.01s avg_case_ms=10.13 simplify=289.85ms avg_simplify_ms=2.90, sum total=200 failed=0 elapsed=886.63ms avg_case_ms=4.43 simplify=286.67ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=628.19ms avg_case_ms=6.28 simplify=181.47ms avg_simplify_ms=1.81, difference total=50 failed=0 elapsed=404.63ms avg_case_ms=8.09 simplify=120.10ms avg_simplify_ms=2.40
- Engine hotspots: shifted_quotient simplify=289.85ms avg_simplify_ms=2.90 wall=1.01s, sum simplify=286.67ms avg_simplify_ms=1.43 wall=886.63ms, product simplify=181.47ms avg_simplify_ms=1.81 wall=628.19ms, difference simplify=120.10ms avg_simplify_ms=2.40 wall=404.63ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.01s avg_case_ms=10.13 avg_simplify_ms=2.90, sum@0+100 failed=0 elapsed=647.47ms avg_case_ms=6.47 avg_simplify_ms=2.00, product@0+100 failed=0 elapsed=628.19ms avg_case_ms=6.28 avg_simplify_ms=1.81, difference@0+50 failed=0 elapsed=404.63ms avg_case_ms=8.09 avg_simplify_ms=2.40, sum@700+100 failed=0 elapsed=239.16ms avg_case_ms=2.39 avg_simplify_ms=0.87
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.92ms median_wire=16.99ms median_wall=64.44ms, difference@0+50 #174 difference runs=3 median_simplify=15.53ms median_wire=15.58ms median_wall=58.48ms, product@0+100 #175 product runs=3 median_simplify=15.76ms median_wire=15.81ms median_wall=59.99ms, sum@0+100 #173 sum runs=3 median_simplify=15.61ms median_wire=15.67ms median_wall=58.78ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.06ms median_wire=13.14ms median_wall=50.34ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.93s | passed=450 failed=0 total=450 avg_case=6.511ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.41s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.76s | passed=1 failed=0 |
