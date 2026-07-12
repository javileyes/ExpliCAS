# Engine Improvement Scorecard

- Generated: 2026-07-12T22:14:18.246305+00:00
- Git branch: main
- Git commit: `1a51c52ffead291cbe37d22e86520655184c34b4`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=953.39ms avg_case_ms=9.53 simplify=266.21ms avg_simplify_ms=2.66, sum total=200 failed=0 elapsed=847.38ms avg_case_ms=4.24 simplify=275.70ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=585.31ms avg_case_ms=5.85 simplify=166.55ms avg_simplify_ms=1.67, difference total=50 failed=0 elapsed=397.74ms avg_case_ms=7.95 simplify=121.13ms avg_simplify_ms=2.42
- Engine hotspots: sum simplify=275.70ms avg_simplify_ms=1.38 wall=847.38ms, shifted_quotient simplify=266.21ms avg_simplify_ms=2.66 wall=953.39ms, product simplify=166.55ms avg_simplify_ms=1.67 wall=585.31ms, difference simplify=121.13ms avg_simplify_ms=2.42 wall=397.74ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=953.39ms avg_case_ms=9.53 avg_simplify_ms=2.66, sum@0+100 failed=0 elapsed=615.51ms avg_case_ms=6.16 avg_simplify_ms=1.92, product@0+100 failed=0 elapsed=585.31ms avg_case_ms=5.85 avg_simplify_ms=1.67, difference@0+50 failed=0 elapsed=397.74ms avg_case_ms=7.95 avg_simplify_ms=2.42, sum@700+100 failed=0 elapsed=231.87ms avg_case_ms=2.32 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.38ms median_wire=16.45ms median_wall=62.86ms, sum@0+100 #173 sum runs=3 median_simplify=15.21ms median_wire=15.25ms median_wall=58.50ms, difference@0+50 #174 difference runs=3 median_simplify=15.20ms median_wire=15.24ms median_wall=57.67ms, product@0+100 #175 product runs=3 median_simplify=14.55ms median_wire=14.60ms median_wall=56.04ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.53ms median_wire=12.59ms median_wall=48.71ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.78s | passed=450 failed=0 total=450 avg_case=6.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.01s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.88s | passed=1 failed=0 |
