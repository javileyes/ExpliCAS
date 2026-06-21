# Engine Improvement Scorecard

- Generated: 2026-06-21T12:14:17.457638+00:00
- Git branch: main
- Git commit: `e83313def68dffaa2c2471b0e4bd698fe34a8f09`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=795.36ms avg_case_ms=7.95 simplify=230.52ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=717.05ms avg_case_ms=3.59 simplify=244.53ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=475.83ms avg_case_ms=4.76 simplify=137.31ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=329.27ms avg_case_ms=6.59 simplify=105.48ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=244.53ms avg_simplify_ms=1.22 wall=717.05ms, shifted_quotient simplify=230.52ms avg_simplify_ms=2.31 wall=795.36ms, product simplify=137.31ms avg_simplify_ms=1.37 wall=475.83ms, difference simplify=105.48ms avg_simplify_ms=2.11 wall=329.27ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=795.36ms avg_case_ms=7.95 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=511.14ms avg_case_ms=5.11 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=475.83ms avg_case_ms=4.76 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=329.27ms avg_case_ms=6.59 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=205.91ms avg_case_ms=2.06 avg_simplify_ms=0.78
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.08ms median_wire=13.15ms median_wall=49.57ms, product@0+100 #175 product runs=3 median_simplify=11.65ms median_wire=11.70ms median_wall=44.45ms, difference@0+50 #174 difference runs=3 median_simplify=12.01ms median_wire=12.07ms median_wall=45.55ms, sum@0+100 #173 sum runs=3 median_simplify=12.15ms median_wire=12.21ms median_wall=45.52ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.78ms median_wire=10.86ms median_wall=40.60ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
