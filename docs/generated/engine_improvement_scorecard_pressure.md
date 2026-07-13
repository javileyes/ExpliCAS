# Engine Improvement Scorecard

- Generated: 2026-07-13T10:46:07.093211+00:00
- Git branch: main
- Git commit: `c3fb5774ef0f232ee4b526d5711d426005805085`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=940.25ms avg_case_ms=9.40 simplify=261.19ms avg_simplify_ms=2.61, sum total=200 failed=0 elapsed=849.57ms avg_case_ms=4.25 simplify=273.48ms avg_simplify_ms=1.37, product total=100 failed=0 elapsed=579.97ms avg_case_ms=5.80 simplify=165.26ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=388.24ms avg_case_ms=7.76 simplify=117.17ms avg_simplify_ms=2.34
- Engine hotspots: sum simplify=273.48ms avg_simplify_ms=1.37 wall=849.57ms, shifted_quotient simplify=261.19ms avg_simplify_ms=2.61 wall=940.25ms, product simplify=165.26ms avg_simplify_ms=1.65 wall=579.97ms, difference simplify=117.17ms avg_simplify_ms=2.34 wall=388.24ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=940.25ms avg_case_ms=9.40 avg_simplify_ms=2.61, sum@0+100 failed=0 elapsed=626.69ms avg_case_ms=6.27 avg_simplify_ms=1.94, product@0+100 failed=0 elapsed=579.97ms avg_case_ms=5.80 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=388.24ms avg_case_ms=7.76 avg_simplify_ms=2.34, sum@700+100 failed=0 elapsed=222.88ms avg_case_ms=2.23 avg_simplify_ms=0.80
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.21ms median_wire=15.26ms median_wall=64.07ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.07ms median_wire=16.13ms median_wall=62.44ms, difference@0+50 #174 difference runs=3 median_simplify=14.95ms median_wire=14.99ms median_wall=57.72ms, product@0+100 #175 product runs=3 median_simplify=15.37ms median_wire=15.43ms median_wall=57.87ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.52ms median_wire=12.60ms median_wall=47.74ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.76s | passed=450 failed=0 total=450 avg_case=6.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.00s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
