# Engine Improvement Scorecard

- Generated: 2026-06-21T18:04:18.908434+00:00
- Git branch: main
- Git commit: `2deeb301c4ad49ec82259e2da2f7045705a818f8`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=790.44ms avg_case_ms=7.90 simplify=226.85ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=699.08ms avg_case_ms=3.50 simplify=237.22ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=480.21ms avg_case_ms=4.80 simplify=138.53ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=332.07ms avg_case_ms=6.64 simplify=105.75ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=237.22ms avg_simplify_ms=1.19 wall=699.08ms, shifted_quotient simplify=226.85ms avg_simplify_ms=2.27 wall=790.44ms, product simplify=138.53ms avg_simplify_ms=1.39 wall=480.21ms, difference simplify=105.75ms avg_simplify_ms=2.11 wall=332.07ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=790.44ms avg_case_ms=7.90 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=507.40ms avg_case_ms=5.07 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=480.21ms avg_case_ms=4.80 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=332.07ms avg_case_ms=6.64 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=191.68ms avg_case_ms=1.92 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.02ms median_wire=13.09ms median_wall=49.48ms, difference@0+50 #174 difference runs=3 median_simplify=11.74ms median_wire=11.81ms median_wall=44.46ms, sum@0+100 #173 sum runs=3 median_simplify=11.68ms median_wire=11.73ms median_wall=44.62ms, product@0+100 #175 product runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.65ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.73ms median_wire=10.80ms median_wall=40.38ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.12s | passed=1 failed=0 |
