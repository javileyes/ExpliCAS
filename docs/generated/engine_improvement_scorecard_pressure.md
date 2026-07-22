# Engine Improvement Scorecard

- Generated: 2026-07-22T14:47:25.056329+00:00
- Git branch: main
- Git commit: `01b38629fcb283f0b747e72f663d5230ba732fe8`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=995.87ms avg_case_ms=9.96 simplify=281.01ms avg_simplify_ms=2.81, sum total=200 failed=0 elapsed=895.59ms avg_case_ms=4.48 simplify=291.43ms avg_simplify_ms=1.46, product total=100 failed=0 elapsed=620.81ms avg_case_ms=6.21 simplify=178.76ms avg_simplify_ms=1.79, difference total=50 failed=0 elapsed=411.73ms avg_case_ms=8.23 simplify=125.59ms avg_simplify_ms=2.51
- Engine hotspots: sum simplify=291.43ms avg_simplify_ms=1.46 wall=895.59ms, shifted_quotient simplify=281.01ms avg_simplify_ms=2.81 wall=995.87ms, product simplify=178.76ms avg_simplify_ms=1.79 wall=620.81ms, difference simplify=125.59ms avg_simplify_ms=2.51 wall=411.73ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=995.87ms avg_case_ms=9.96 avg_simplify_ms=2.81, sum@0+100 failed=0 elapsed=657.92ms avg_case_ms=6.58 avg_simplify_ms=2.06, product@0+100 failed=0 elapsed=620.81ms avg_case_ms=6.21 avg_simplify_ms=1.79, difference@0+50 failed=0 elapsed=411.73ms avg_case_ms=8.23 avg_simplify_ms=2.51, sum@700+100 failed=0 elapsed=237.67ms avg_case_ms=2.38 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.26ms median_wire=17.33ms median_wall=67.04ms, sum@0+100 #173 sum runs=3 median_simplify=22.34ms median_wire=22.49ms median_wall=70.32ms, difference@0+50 #174 difference runs=3 median_simplify=15.48ms median_wire=15.53ms median_wall=65.11ms, product@0+100 #175 product runs=3 median_simplify=15.82ms median_wire=15.88ms median_wall=59.25ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.42ms median_wire=13.49ms median_wall=50.58ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.93s | passed=450 failed=0 total=450 avg_case=6.511ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.66s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
