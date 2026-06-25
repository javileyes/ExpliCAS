# Engine Improvement Scorecard

- Generated: 2026-06-25T21:39:21.920564+00:00
- Git branch: main
- Git commit: `9514c5ab2c25807f553ef44b056d5c0109873859`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=791.67ms avg_case_ms=7.92 simplify=226.17ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=704.15ms avg_case_ms=3.52 simplify=243.22ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=485.08ms avg_case_ms=4.85 simplify=142.52ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=329.73ms avg_case_ms=6.59 simplify=105.84ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=243.22ms avg_simplify_ms=1.22 wall=704.15ms, shifted_quotient simplify=226.17ms avg_simplify_ms=2.26 wall=791.67ms, product simplify=142.52ms avg_simplify_ms=1.43 wall=485.08ms, difference simplify=105.84ms avg_simplify_ms=2.12 wall=329.73ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=791.67ms avg_case_ms=7.92 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=507.71ms avg_case_ms=5.08 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=485.08ms avg_case_ms=4.85 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=329.73ms avg_case_ms=6.59 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=196.44ms avg_case_ms=1.96 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.90ms median_wire=12.97ms median_wall=49.06ms, difference@0+50 #174 difference runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=44.25ms, sum@0+100 #173 sum runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.79ms, product@0+100 #175 product runs=3 median_simplify=12.04ms median_wire=12.10ms median_wall=45.86ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.53ms median_wire=10.60ms median_wall=39.85ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.47s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
