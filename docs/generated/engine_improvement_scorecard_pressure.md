# Engine Improvement Scorecard

- Generated: 2026-07-08T07:54:37.375204+00:00
- Git branch: main
- Git commit: `1e99823c8732c6efcf0a2757110be0b0b1e3dd22`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=959.96ms avg_case_ms=9.60 simplify=266.48ms avg_simplify_ms=2.66, sum total=200 failed=0 elapsed=841.73ms avg_case_ms=4.21 simplify=271.92ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=595.33ms avg_case_ms=5.95 simplify=171.32ms avg_simplify_ms=1.71, difference total=50 failed=0 elapsed=393.96ms avg_case_ms=7.88 simplify=119.12ms avg_simplify_ms=2.38
- Engine hotspots: sum simplify=271.92ms avg_simplify_ms=1.36 wall=841.73ms, shifted_quotient simplify=266.48ms avg_simplify_ms=2.66 wall=959.96ms, product simplify=171.32ms avg_simplify_ms=1.71 wall=595.33ms, difference simplify=119.12ms avg_simplify_ms=2.38 wall=393.96ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=959.96ms avg_case_ms=9.60 avg_simplify_ms=2.66, sum@0+100 failed=0 elapsed=612.84ms avg_case_ms=6.13 avg_simplify_ms=1.91, product@0+100 failed=0 elapsed=595.33ms avg_case_ms=5.95 avg_simplify_ms=1.71, difference@0+50 failed=0 elapsed=393.96ms avg_case_ms=7.88 avg_simplify_ms=2.38, sum@700+100 failed=0 elapsed=228.89ms avg_case_ms=2.29 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.63ms median_wire=16.70ms median_wall=64.16ms, product@0+100 #175 product runs=3 median_simplify=14.60ms median_wire=14.65ms median_wall=56.48ms, sum@0+100 #173 sum runs=3 median_simplify=15.23ms median_wire=15.29ms median_wall=58.29ms, difference@0+50 #174 difference runs=3 median_simplify=14.99ms median_wire=15.04ms median_wall=57.18ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.57ms median_wire=12.64ms median_wall=47.99ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.79s | passed=450 failed=0 total=450 avg_case=6.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.05s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.88s | passed=1 failed=0 |
