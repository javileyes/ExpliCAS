# Engine Improvement Scorecard

- Generated: 2026-06-21T20:45:59.259158+00:00
- Git branch: main
- Git commit: `e12a8880c9f129ddc8462402c835b8465e6979ed`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=792.08ms avg_case_ms=7.92 simplify=226.62ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=720.65ms avg_case_ms=3.60 simplify=245.79ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=495.34ms avg_case_ms=4.95 simplify=143.65ms avg_simplify_ms=1.44, difference total=50 failed=0 elapsed=344.04ms avg_case_ms=6.88 simplify=108.68ms avg_simplify_ms=2.17
- Engine hotspots: sum simplify=245.79ms avg_simplify_ms=1.23 wall=720.65ms, shifted_quotient simplify=226.62ms avg_simplify_ms=2.27 wall=792.08ms, product simplify=143.65ms avg_simplify_ms=1.44 wall=495.34ms, difference simplify=108.68ms avg_simplify_ms=2.17 wall=344.04ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=792.08ms avg_case_ms=7.92 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=507.89ms avg_case_ms=5.08 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=495.34ms avg_case_ms=4.95 avg_simplify_ms=1.44, difference@0+50 failed=0 elapsed=344.04ms avg_case_ms=6.88 avg_simplify_ms=2.17, sum@700+100 failed=0 elapsed=212.76ms avg_case_ms=2.13 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.12ms median_wire=13.19ms median_wall=49.78ms, difference@0+50 #174 difference runs=3 median_simplify=11.76ms median_wire=11.81ms median_wall=44.81ms, product@0+100 #175 product runs=3 median_simplify=11.75ms median_wire=11.81ms median_wall=44.88ms, sum@0+100 #173 sum runs=3 median_simplify=12.05ms median_wire=12.10ms median_wall=45.37ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.71ms median_wire=10.79ms median_wall=40.36ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.35s | passed=450 failed=0 total=450 avg_case=5.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
