# Engine Improvement Scorecard

- Generated: 2026-07-18T20:44:04.259052+00:00
- Git branch: main
- Git commit: `9df0aa329ea67a279f57db3a032d8754a6193eee`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.02s avg_case_ms=10.16 simplify=290.32ms avg_simplify_ms=2.90, sum total=200 failed=0 elapsed=903.99ms avg_case_ms=4.52 simplify=294.24ms avg_simplify_ms=1.47, product total=100 failed=0 elapsed=636.01ms avg_case_ms=6.36 simplify=183.57ms avg_simplify_ms=1.84, difference total=50 failed=0 elapsed=418.48ms avg_case_ms=8.37 simplify=128.50ms avg_simplify_ms=2.57
- Engine hotspots: sum simplify=294.24ms avg_simplify_ms=1.47 wall=903.99ms, shifted_quotient simplify=290.32ms avg_simplify_ms=2.90 wall=1.02s, product simplify=183.57ms avg_simplify_ms=1.84 wall=636.01ms, difference simplify=128.50ms avg_simplify_ms=2.57 wall=418.48ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.02s avg_case_ms=10.16 avg_simplify_ms=2.90, sum@0+100 failed=0 elapsed=663.79ms avg_case_ms=6.64 avg_simplify_ms=2.08, product@0+100 failed=0 elapsed=636.01ms avg_case_ms=6.36 avg_simplify_ms=1.84, difference@0+50 failed=0 elapsed=418.48ms avg_case_ms=8.37 avg_simplify_ms=2.57, sum@700+100 failed=0 elapsed=240.20ms avg_case_ms=2.40 avg_simplify_ms=0.87
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.67ms median_wire=15.72ms median_wall=59.38ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.22ms median_wire=17.30ms median_wall=66.06ms, difference@0+50 #174 difference runs=3 median_simplify=15.60ms median_wire=15.66ms median_wall=60.00ms, product@0+100 #175 product runs=3 median_simplify=15.45ms median_wire=15.51ms median_wall=59.36ms, shifted_quotient@0+100 #112 shifted_quotient runs=3 median_simplify=9.90ms median_wire=9.96ms median_wall=37.11ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.98s | passed=450 failed=0 total=450 avg_case=6.622ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.88s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.97s | passed=1 failed=0 |
