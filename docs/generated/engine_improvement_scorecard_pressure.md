# Engine Improvement Scorecard

- Generated: 2026-06-26T16:25:05.863686+00:00
- Git branch: main
- Git commit: `f389b62627c0dd87032ab76d66c67c41168c3801`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=793.46ms avg_case_ms=7.93 simplify=227.76ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=710.68ms avg_case_ms=3.55 simplify=245.95ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=481.98ms avg_case_ms=4.82 simplify=141.57ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=328.60ms avg_case_ms=6.57 simplify=106.44ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=245.95ms avg_simplify_ms=1.23 wall=710.68ms, shifted_quotient simplify=227.76ms avg_simplify_ms=2.28 wall=793.46ms, product simplify=141.57ms avg_simplify_ms=1.42 wall=481.98ms, difference simplify=106.44ms avg_simplify_ms=2.13 wall=328.60ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=793.46ms avg_case_ms=7.93 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=511.16ms avg_case_ms=5.11 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=481.98ms avg_case_ms=4.82 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=328.60ms avg_case_ms=6.57 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=199.52ms avg_case_ms=2.00 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.91ms median_wire=12.98ms median_wall=49.34ms, product@0+100 #175 product runs=3 median_simplify=11.80ms median_wire=11.85ms median_wall=44.79ms, difference@0+50 #174 difference runs=3 median_simplify=11.83ms median_wire=11.88ms median_wall=44.98ms, sum@0+100 #173 sum runs=3 median_simplify=11.84ms median_wire=11.89ms median_wall=44.39ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.76ms median_wire=10.83ms median_wall=40.26ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
