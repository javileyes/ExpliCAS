# Engine Improvement Scorecard

- Generated: 2026-06-27T20:14:53.817795+00:00
- Git branch: main
- Git commit: `1b9a533297ebb22dc75ea465e2e44851aa879de9`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=793.94ms avg_case_ms=7.94 simplify=225.21ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=717.03ms avg_case_ms=3.59 simplify=246.40ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=487.72ms avg_case_ms=4.88 simplify=144.21ms avg_simplify_ms=1.44, difference total=50 failed=0 elapsed=334.86ms avg_case_ms=6.70 simplify=108.41ms avg_simplify_ms=2.17
- Engine hotspots: sum simplify=246.40ms avg_simplify_ms=1.23 wall=717.03ms, shifted_quotient simplify=225.21ms avg_simplify_ms=2.25 wall=793.94ms, product simplify=144.21ms avg_simplify_ms=1.44 wall=487.72ms, difference simplify=108.41ms avg_simplify_ms=2.17 wall=334.86ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=793.94ms avg_case_ms=7.94 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=516.82ms avg_case_ms=5.17 avg_simplify_ms=1.70, product@0+100 failed=0 elapsed=487.72ms avg_case_ms=4.88 avg_simplify_ms=1.44, difference@0+50 failed=0 elapsed=334.86ms avg_case_ms=6.70 avg_simplify_ms=2.17, sum@700+100 failed=0 elapsed=200.21ms avg_case_ms=2.00 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.99ms median_wire=13.06ms median_wall=49.88ms, difference@0+50 #174 difference runs=3 median_simplify=11.80ms median_wire=11.85ms median_wall=44.98ms, sum@0+100 #173 sum runs=3 median_simplify=12.42ms median_wire=12.47ms median_wall=47.41ms, product@0+100 #175 product runs=3 median_simplify=13.05ms median_wire=13.11ms median_wall=46.08ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.79ms median_wire=10.87ms median_wall=40.70ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
