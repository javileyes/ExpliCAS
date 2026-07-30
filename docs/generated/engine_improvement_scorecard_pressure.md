# Engine Improvement Scorecard

- Generated: 2026-07-30T23:25:43.953572+00:00
- Git branch: main
- Git commit: `3aacc442d43c8d6fbc85547e5de30c6e3293868d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=966.61ms avg_case_ms=9.67 simplify=270.24ms avg_simplify_ms=2.70, sum total=200 failed=0 elapsed=839.95ms avg_case_ms=4.20 simplify=269.08ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=610.54ms avg_case_ms=6.11 simplify=174.96ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=390.78ms avg_case_ms=7.82 simplify=115.15ms avg_simplify_ms=2.30
- Engine hotspots: shifted_quotient simplify=270.24ms avg_simplify_ms=2.70 wall=966.61ms, sum simplify=269.08ms avg_simplify_ms=1.35 wall=839.95ms, product simplify=174.96ms avg_simplify_ms=1.75 wall=610.54ms, difference simplify=115.15ms avg_simplify_ms=2.30 wall=390.78ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=966.61ms avg_case_ms=9.67 avg_simplify_ms=2.70, sum@0+100 failed=0 elapsed=613.21ms avg_case_ms=6.13 avg_simplify_ms=1.88, product@0+100 failed=0 elapsed=610.54ms avg_case_ms=6.11 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=390.78ms avg_case_ms=7.82 avg_simplify_ms=2.30, sum@700+100 failed=0 elapsed=226.74ms avg_case_ms=2.27 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.80ms median_wire=16.87ms median_wall=64.59ms, difference@0+50 #174 difference runs=3 median_simplify=15.29ms median_wire=15.34ms median_wall=58.10ms, sum@0+100 #173 sum runs=3 median_simplify=15.33ms median_wire=15.38ms median_wall=58.20ms, product@0+100 #175 product runs=3 median_simplify=15.00ms median_wire=15.05ms median_wall=56.87ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.91ms median_wire=12.98ms median_wall=48.68ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.81s | passed=450 failed=0 total=450 avg_case=6.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.20s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.74s | passed=1 failed=0 |
