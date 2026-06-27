# Engine Improvement Scorecard

- Generated: 2026-06-27T00:35:40.889347+00:00
- Git branch: main
- Git commit: `b9bd403afc43cb0ec4190f6e250447e74427032d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=793.44ms avg_case_ms=7.93 simplify=227.51ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=703.77ms avg_case_ms=3.52 simplify=243.50ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=488.42ms avg_case_ms=4.88 simplify=143.25ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=334.68ms avg_case_ms=6.69 simplify=109.76ms avg_simplify_ms=2.20
- Engine hotspots: sum simplify=243.50ms avg_simplify_ms=1.22 wall=703.77ms, shifted_quotient simplify=227.51ms avg_simplify_ms=2.28 wall=793.44ms, product simplify=143.25ms avg_simplify_ms=1.43 wall=488.42ms, difference simplify=109.76ms avg_simplify_ms=2.20 wall=334.68ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=793.44ms avg_case_ms=7.93 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=506.24ms avg_case_ms=5.06 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=488.42ms avg_case_ms=4.88 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=334.68ms avg_case_ms=6.69 avg_simplify_ms=2.20, sum@700+100 failed=0 elapsed=197.54ms avg_case_ms=1.98 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.98ms median_wire=13.05ms median_wall=49.89ms, product@0+100 #175 product runs=3 median_simplify=11.71ms median_wire=11.76ms median_wall=44.69ms, sum@0+100 #173 sum runs=3 median_simplify=11.84ms median_wire=11.89ms median_wall=44.75ms, difference@0+50 #174 difference runs=3 median_simplify=11.51ms median_wire=11.56ms median_wall=44.67ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.80ms median_wire=10.88ms median_wall=40.50ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
