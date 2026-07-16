# Engine Improvement Scorecard

- Generated: 2026-07-16T23:12:05.528130+00:00
- Git branch: main
- Git commit: `561efcb6e421d68df6653b4cb7d078e85eea54fc`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.03s avg_case_ms=10.27 simplify=286.57ms avg_simplify_ms=2.87, sum total=200 failed=0 elapsed=911.25ms avg_case_ms=4.56 simplify=298.67ms avg_simplify_ms=1.49, product total=100 failed=0 elapsed=622.01ms avg_case_ms=6.22 simplify=179.46ms avg_simplify_ms=1.79, difference total=50 failed=0 elapsed=414.48ms avg_case_ms=8.29 simplify=125.84ms avg_simplify_ms=2.52
- Engine hotspots: sum simplify=298.67ms avg_simplify_ms=1.49 wall=911.25ms, shifted_quotient simplify=286.57ms avg_simplify_ms=2.87 wall=1.03s, product simplify=179.46ms avg_simplify_ms=1.79 wall=622.01ms, difference simplify=125.84ms avg_simplify_ms=2.52 wall=414.48ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.03s avg_case_ms=10.27 avg_simplify_ms=2.87, sum@0+100 failed=0 elapsed=675.07ms avg_case_ms=6.75 avg_simplify_ms=2.14, product@0+100 failed=0 elapsed=622.01ms avg_case_ms=6.22 avg_simplify_ms=1.79, difference@0+50 failed=0 elapsed=414.48ms avg_case_ms=8.29 avg_simplify_ms=2.52, sum@700+100 failed=0 elapsed=236.18ms avg_case_ms=2.36 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.85ms median_wire=16.92ms median_wall=64.76ms, sum@0+100 #157 sum runs=3 median_simplify=10.01ms median_wire=10.08ms median_wall=37.10ms, sum@0+100 #173 sum runs=3 median_simplify=15.30ms median_wire=15.35ms median_wall=58.59ms, difference@0+50 #174 difference runs=3 median_simplify=15.64ms median_wire=15.70ms median_wall=70.13ms, product@0+100 #175 product runs=3 median_simplify=15.31ms median_wire=15.36ms median_wall=59.33ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #157 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^3) + ln(y^2) - ln(x^3 * y^2)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.98s | passed=450 failed=0 total=450 avg_case=6.622ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.96s | passed=1 failed=0 |
