# Engine Improvement Scorecard

- Generated: 2026-06-26T23:27:53.126804+00:00
- Git branch: main
- Git commit: `18c17941dc10858641981c2bf5046caa39905a84`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=791.26ms avg_case_ms=7.91 simplify=226.11ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=707.78ms avg_case_ms=3.54 simplify=244.24ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=487.71ms avg_case_ms=4.88 simplify=143.47ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=327.16ms avg_case_ms=6.54 simplify=106.05ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=244.24ms avg_simplify_ms=1.22 wall=707.78ms, shifted_quotient simplify=226.11ms avg_simplify_ms=2.26 wall=791.26ms, product simplify=143.47ms avg_simplify_ms=1.43 wall=487.71ms, difference simplify=106.05ms avg_simplify_ms=2.12 wall=327.16ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=791.26ms avg_case_ms=7.91 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=512.08ms avg_case_ms=5.12 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=487.71ms avg_case_ms=4.88 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=327.16ms avg_case_ms=6.54 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=195.71ms avg_case_ms=1.96 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.23ms median_wire=13.30ms median_wall=50.09ms, difference@0+50 #174 difference runs=3 median_simplify=11.82ms median_wire=11.88ms median_wall=45.25ms, sum@0+100 #173 sum runs=3 median_simplify=12.16ms median_wire=12.21ms median_wall=45.29ms, product@0+100 #175 product runs=3 median_simplify=11.65ms median_wire=11.71ms median_wall=44.52ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.86ms median_wire=10.93ms median_wall=40.71ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
