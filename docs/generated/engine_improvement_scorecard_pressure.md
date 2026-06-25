# Engine Improvement Scorecard

- Generated: 2026-06-25T06:58:55.590029+00:00
- Git branch: main
- Git commit: `22b55ea39d37972151a6be0b6a025506b876b09e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=796.99ms avg_case_ms=7.97 simplify=229.50ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=711.66ms avg_case_ms=3.56 simplify=243.87ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=487.70ms avg_case_ms=4.88 simplify=141.42ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=330.20ms avg_case_ms=6.60 simplify=105.89ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=243.87ms avg_simplify_ms=1.22 wall=711.66ms, shifted_quotient simplify=229.50ms avg_simplify_ms=2.30 wall=796.99ms, product simplify=141.42ms avg_simplify_ms=1.41 wall=487.70ms, difference simplify=105.89ms avg_simplify_ms=2.12 wall=330.20ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=796.99ms avg_case_ms=7.97 avg_simplify_ms=2.30, sum@0+100 failed=0 elapsed=514.14ms avg_case_ms=5.14 avg_simplify_ms=1.70, product@0+100 failed=0 elapsed=487.70ms avg_case_ms=4.88 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=330.20ms avg_case_ms=6.60 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=197.52ms avg_case_ms=1.98 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.11ms median_wire=13.18ms median_wall=49.66ms, sum@0+100 #173 sum runs=3 median_simplify=11.87ms median_wire=11.93ms median_wall=44.78ms, product@0+100 #175 product runs=3 median_simplify=11.79ms median_wire=11.84ms median_wall=44.89ms, difference@0+50 #174 difference runs=3 median_simplify=11.64ms median_wire=11.69ms median_wall=44.62ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.69ms median_wire=10.76ms median_wall=40.58ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.47s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
