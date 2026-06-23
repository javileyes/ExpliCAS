# Engine Improvement Scorecard

- Generated: 2026-06-23T23:03:52.648474+00:00
- Git branch: main
- Git commit: `c6960a04aac04a913ddaf02689d237fa6081e6c6`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=786.96ms avg_case_ms=7.87 simplify=224.67ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=696.99ms avg_case_ms=3.48 simplify=236.47ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=479.00ms avg_case_ms=4.79 simplify=138.05ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=328.96ms avg_case_ms=6.58 simplify=105.81ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=236.47ms avg_simplify_ms=1.18 wall=696.99ms, shifted_quotient simplify=224.67ms avg_simplify_ms=2.25 wall=786.96ms, product simplify=138.05ms avg_simplify_ms=1.38 wall=479.00ms, difference simplify=105.81ms avg_simplify_ms=2.12 wall=328.96ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=786.96ms avg_case_ms=7.87 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=506.06ms avg_case_ms=5.06 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=479.00ms avg_case_ms=4.79 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=328.96ms avg_case_ms=6.58 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=190.93ms avg_case_ms=1.91 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.99ms median_wire=13.06ms median_wall=49.59ms, sum@0+100 #173 sum runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=44.50ms, difference@0+50 #174 difference runs=3 median_simplify=11.91ms median_wire=11.97ms median_wall=44.85ms, product@0+100 #175 product runs=3 median_simplify=11.87ms median_wire=11.92ms median_wall=44.72ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.66ms median_wire=10.74ms median_wall=40.49ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
