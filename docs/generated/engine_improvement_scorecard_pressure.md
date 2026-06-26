# Engine Improvement Scorecard

- Generated: 2026-06-26T20:42:38.231354+00:00
- Git branch: main
- Git commit: `f6ba0314ec5c07e6a273c6dc5d680195f60d60ed`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=788.32ms avg_case_ms=7.88 simplify=225.64ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=717.51ms avg_case_ms=3.59 simplify=247.77ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=484.20ms avg_case_ms=4.84 simplify=141.57ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=331.39ms avg_case_ms=6.63 simplify=106.59ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=247.77ms avg_simplify_ms=1.24 wall=717.51ms, shifted_quotient simplify=225.64ms avg_simplify_ms=2.26 wall=788.32ms, product simplify=141.57ms avg_simplify_ms=1.42 wall=484.20ms, difference simplify=106.59ms avg_simplify_ms=2.13 wall=331.39ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=788.32ms avg_case_ms=7.88 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=506.90ms avg_case_ms=5.07 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=484.20ms avg_case_ms=4.84 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=331.39ms avg_case_ms=6.63 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=210.61ms avg_case_ms=2.11 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.07ms median_wire=13.14ms median_wall=49.73ms, difference@0+50 #174 difference runs=3 median_simplify=11.79ms median_wire=11.85ms median_wall=44.77ms, sum@0+100 #173 sum runs=3 median_simplify=11.83ms median_wire=11.89ms median_wall=44.28ms, product@0+100 #175 product runs=3 median_simplify=11.68ms median_wire=11.73ms median_wall=44.21ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.53ms median_wire=10.61ms median_wall=40.22ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
