# Engine Improvement Scorecard

- Generated: 2026-06-26T14:54:44.719001+00:00
- Git branch: main
- Git commit: `924c2c9236db290638bc614b4b9f3e1f66b16e8d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=804.83ms avg_case_ms=8.05 simplify=230.92ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=717.36ms avg_case_ms=3.59 simplify=248.33ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=496.73ms avg_case_ms=4.97 simplify=146.19ms avg_simplify_ms=1.46, difference total=50 failed=0 elapsed=331.35ms avg_case_ms=6.63 simplify=106.53ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=248.33ms avg_simplify_ms=1.24 wall=717.36ms, shifted_quotient simplify=230.92ms avg_simplify_ms=2.31 wall=804.83ms, product simplify=146.19ms avg_simplify_ms=1.46 wall=496.73ms, difference simplify=106.53ms avg_simplify_ms=2.13 wall=331.35ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=804.83ms avg_case_ms=8.05 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=509.42ms avg_case_ms=5.09 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=496.73ms avg_case_ms=4.97 avg_simplify_ms=1.46, difference@0+50 failed=0 elapsed=331.35ms avg_case_ms=6.63 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=207.94ms avg_case_ms=2.08 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.38ms median_wire=13.46ms median_wall=50.65ms, difference@0+50 #174 difference runs=3 median_simplify=11.73ms median_wire=11.79ms median_wall=44.55ms, sum@0+100 #173 sum runs=3 median_simplify=11.98ms median_wire=12.03ms median_wall=46.77ms, product@0+100 #175 product runs=3 median_simplify=11.86ms median_wire=11.91ms median_wall=45.33ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.71ms median_wire=10.79ms median_wall=40.33ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.35s | passed=450 failed=0 total=450 avg_case=5.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.12s | passed=1 failed=0 |
