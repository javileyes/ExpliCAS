# Engine Improvement Scorecard

- Generated: 2026-06-14T11:52:23.448370+00:00
- Git branch: main
- Git commit: `895440035d6e9c5d92b5cc3a4bb4ecf6617bcc87`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=345

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=761.59ms avg_case_ms=7.62 simplify=216.15ms avg_simplify_ms=2.16, sum total=200 failed=0 elapsed=685.25ms avg_case_ms=3.43 simplify=228.39ms avg_simplify_ms=1.14, product total=100 failed=0 elapsed=462.51ms avg_case_ms=4.63 simplify=132.05ms avg_simplify_ms=1.32, difference total=50 failed=0 elapsed=319.60ms avg_case_ms=6.39 simplify=100.53ms avg_simplify_ms=2.01
- Engine hotspots: sum simplify=228.39ms avg_simplify_ms=1.14 wall=685.25ms, shifted_quotient simplify=216.15ms avg_simplify_ms=2.16 wall=761.59ms, product simplify=132.05ms avg_simplify_ms=1.32 wall=462.51ms, difference simplify=100.53ms avg_simplify_ms=2.01 wall=319.60ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=761.59ms avg_case_ms=7.62 avg_simplify_ms=2.16, sum@0+100 failed=0 elapsed=497.82ms avg_case_ms=4.98 avg_simplify_ms=1.59, product@0+100 failed=0 elapsed=462.51ms avg_case_ms=4.63 avg_simplify_ms=1.32, difference@0+50 failed=0 elapsed=319.60ms avg_case_ms=6.39 avg_simplify_ms=2.01, sum@700+100 failed=0 elapsed=187.43ms avg_case_ms=1.87 avg_simplify_ms=0.69
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.89ms median_wire=12.97ms median_wall=48.88ms, sum@0+100 #173 sum runs=3 median_simplify=11.43ms median_wire=11.47ms median_wall=43.63ms, difference@0+50 #174 difference runs=3 median_simplify=11.44ms median_wire=11.49ms median_wall=43.37ms, product@0+100 #175 product runs=3 median_simplify=11.44ms median_wire=11.49ms median_wall=43.78ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.03ms median_wire=10.09ms median_wall=38.46ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.23s | passed=450 failed=0 total=450 avg_case=4.956ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
