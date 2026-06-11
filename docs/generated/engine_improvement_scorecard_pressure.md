# Engine Improvement Scorecard

- Generated: 2026-06-11T17:39:39.325530+00:00
- Git branch: main
- Git commit: `981b34845b4a8baa5d01ff06f25c055549be6a04`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=784.71ms avg_case_ms=7.85 simplify=221.34ms avg_simplify_ms=2.21, sum total=200 failed=0 elapsed=699.68ms avg_case_ms=3.50 simplify=231.76ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=475.07ms avg_case_ms=4.75 simplify=135.73ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=322.20ms avg_case_ms=6.44 simplify=102.18ms avg_simplify_ms=2.04
- Engine hotspots: sum simplify=231.76ms avg_simplify_ms=1.16 wall=699.68ms, shifted_quotient simplify=221.34ms avg_simplify_ms=2.21 wall=784.71ms, product simplify=135.73ms avg_simplify_ms=1.36 wall=475.07ms, difference simplify=102.18ms avg_simplify_ms=2.04 wall=322.20ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=784.71ms avg_case_ms=7.85 avg_simplify_ms=2.21, sum@0+100 failed=0 elapsed=510.85ms avg_case_ms=5.11 avg_simplify_ms=1.62, product@0+100 failed=0 elapsed=475.07ms avg_case_ms=4.75 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=322.20ms avg_case_ms=6.44 avg_simplify_ms=2.04, sum@700+100 failed=0 elapsed=188.82ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=14.17ms median_wire=14.24ms median_wall=49.82ms, difference@0+50 #174 difference runs=3 median_simplify=11.61ms median_wire=11.66ms median_wall=43.94ms, sum@0+100 #173 sum runs=3 median_simplify=11.54ms median_wire=11.59ms median_wall=43.81ms, product@0+100 #175 product runs=3 median_simplify=11.54ms median_wire=11.59ms median_wall=44.25ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.95ms median_wire=11.03ms median_wall=41.44ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.85s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.06s | passed=1 failed=0 |
