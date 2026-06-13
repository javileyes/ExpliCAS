# Engine Improvement Scorecard

- Generated: 2026-06-13T13:17:40.475570+00:00
- Git branch: main
- Git commit: `bd00c7a75ace1326b9d4b4f96c9b985285978880`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=782.35ms avg_case_ms=7.82 simplify=222.51ms avg_simplify_ms=2.23, sum total=200 failed=0 elapsed=683.51ms avg_case_ms=3.42 simplify=229.03ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=473.27ms avg_case_ms=4.73 simplify=135.52ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=327.64ms avg_case_ms=6.55 simplify=103.75ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=229.03ms avg_simplify_ms=1.15 wall=683.51ms, shifted_quotient simplify=222.51ms avg_simplify_ms=2.23 wall=782.35ms, product simplify=135.52ms avg_simplify_ms=1.36 wall=473.27ms, difference simplify=103.75ms avg_simplify_ms=2.07 wall=327.64ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=782.35ms avg_case_ms=7.82 avg_simplify_ms=2.23, sum@0+100 failed=0 elapsed=495.85ms avg_case_ms=4.96 avg_simplify_ms=1.59, product@0+100 failed=0 elapsed=473.27ms avg_case_ms=4.73 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=327.64ms avg_case_ms=6.55 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=187.66ms avg_case_ms=1.88 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.82ms median_wire=12.89ms median_wall=48.52ms, difference@0+50 #174 difference runs=3 median_simplify=11.53ms median_wire=11.58ms median_wall=43.62ms, sum@0+100 #173 sum runs=3 median_simplify=11.50ms median_wire=11.55ms median_wall=43.84ms, product@0+100 #175 product runs=3 median_simplify=11.61ms median_wire=11.65ms median_wall=44.08ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.61ms median_wire=10.68ms median_wall=40.21ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.94s | passed=1 failed=0 |
