# Engine Improvement Scorecard

- Generated: 2026-06-18T16:44:04.396284+00:00
- Git branch: main
- Git commit: `4caffba9394b48e21da31c187b811113c6893112`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=798.66ms avg_case_ms=7.99 simplify=229.02ms avg_simplify_ms=2.29, sum total=200 failed=0 elapsed=703.65ms avg_case_ms=3.52 simplify=238.67ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=487.68ms avg_case_ms=4.88 simplify=141.25ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=328.00ms avg_case_ms=6.56 simplify=105.11ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=238.67ms avg_simplify_ms=1.19 wall=703.65ms, shifted_quotient simplify=229.02ms avg_simplify_ms=2.29 wall=798.66ms, product simplify=141.25ms avg_simplify_ms=1.41 wall=487.68ms, difference simplify=105.11ms avg_simplify_ms=2.10 wall=328.00ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=798.66ms avg_case_ms=7.99 avg_simplify_ms=2.29, sum@0+100 failed=0 elapsed=511.55ms avg_case_ms=5.12 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=487.68ms avg_case_ms=4.88 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=328.00ms avg_case_ms=6.56 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=192.10ms avg_case_ms=1.92 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.25ms median_wire=13.32ms median_wall=50.12ms, difference@0+50 #174 difference runs=3 median_simplify=11.81ms median_wire=11.86ms median_wall=44.92ms, sum@0+100 #173 sum runs=3 median_simplify=11.85ms median_wire=11.90ms median_wall=45.10ms, product@0+100 #175 product runs=3 median_simplify=12.04ms median_wire=12.10ms median_wall=45.34ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.94ms median_wire=11.03ms median_wall=40.94ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.49s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
