# Engine Improvement Scorecard

- Generated: 2026-07-02T03:12:18.059324+00:00
- Git branch: main
- Git commit: `313674b1da2485e27afa86d9813ca15a08d3d2ef`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=355

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=806.89ms avg_case_ms=8.07 simplify=234.07ms avg_simplify_ms=2.34, sum total=200 failed=0 elapsed=723.80ms avg_case_ms=3.62 simplify=250.67ms avg_simplify_ms=1.25, product total=100 failed=0 elapsed=496.93ms avg_case_ms=4.97 simplify=148.29ms avg_simplify_ms=1.48, difference total=50 failed=0 elapsed=331.34ms avg_case_ms=6.63 simplify=107.11ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=250.67ms avg_simplify_ms=1.25 wall=723.80ms, shifted_quotient simplify=234.07ms avg_simplify_ms=2.34 wall=806.89ms, product simplify=148.29ms avg_simplify_ms=1.48 wall=496.93ms, difference simplify=107.11ms avg_simplify_ms=2.14 wall=331.34ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=806.89ms avg_case_ms=8.07 avg_simplify_ms=2.34, sum@0+100 failed=0 elapsed=526.87ms avg_case_ms=5.27 avg_simplify_ms=1.75, product@0+100 failed=0 elapsed=496.93ms avg_case_ms=4.97 avg_simplify_ms=1.48, difference@0+50 failed=0 elapsed=331.34ms avg_case_ms=6.63 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=196.93ms avg_case_ms=1.97 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.08ms median_wire=13.15ms median_wall=49.82ms, sum@0+100 #173 sum runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.93ms, difference@0+50 #174 difference runs=3 median_simplify=11.75ms median_wire=11.79ms median_wall=44.06ms, product@0+100 #175 product runs=3 median_simplify=11.82ms median_wire=11.87ms median_wall=44.48ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.35ms median_wire=11.42ms median_wall=42.12ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.36s | passed=450 failed=0 total=450 avg_case=5.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.47s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
