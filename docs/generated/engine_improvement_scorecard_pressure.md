# Engine Improvement Scorecard

- Generated: 2026-06-24T15:14:45.339157+00:00
- Git branch: main
- Git commit: `3eeff717d66db98d7f7b25a7fc557ef610ff64ff`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=793.33ms avg_case_ms=7.93 simplify=227.72ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=701.21ms avg_case_ms=3.51 simplify=237.64ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=480.37ms avg_case_ms=4.80 simplify=138.59ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=330.31ms avg_case_ms=6.61 simplify=106.04ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=237.64ms avg_simplify_ms=1.19 wall=701.21ms, shifted_quotient simplify=227.72ms avg_simplify_ms=2.28 wall=793.33ms, product simplify=138.59ms avg_simplify_ms=1.39 wall=480.37ms, difference simplify=106.04ms avg_simplify_ms=2.12 wall=330.31ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=793.33ms avg_case_ms=7.93 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=508.15ms avg_case_ms=5.08 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=480.37ms avg_case_ms=4.80 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=330.31ms avg_case_ms=6.61 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=193.06ms avg_case_ms=1.93 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.15ms median_wire=13.23ms median_wall=50.34ms, difference@0+50 #174 difference runs=3 median_simplify=11.78ms median_wire=11.83ms median_wall=44.52ms, product@0+100 #175 product runs=3 median_simplify=11.95ms median_wire=12.01ms median_wall=45.39ms, sum@0+100 #173 sum runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.51ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.72ms median_wire=10.79ms median_wall=40.73ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
