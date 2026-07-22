# Engine Improvement Scorecard

- Generated: 2026-07-22T23:00:37.423065+00:00
- Git branch: main
- Git commit: `ed0d15d4e39393abce4aa79e3f331fd60a518863`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=969.30ms avg_case_ms=9.69 simplify=271.02ms avg_simplify_ms=2.71, sum total=200 failed=0 elapsed=901.59ms avg_case_ms=4.51 simplify=291.54ms avg_simplify_ms=1.46, product total=100 failed=0 elapsed=601.58ms avg_case_ms=6.02 simplify=173.20ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=396.15ms avg_case_ms=7.92 simplify=120.91ms avg_simplify_ms=2.42
- Engine hotspots: sum simplify=291.54ms avg_simplify_ms=1.46 wall=901.59ms, shifted_quotient simplify=271.02ms avg_simplify_ms=2.71 wall=969.30ms, product simplify=173.20ms avg_simplify_ms=1.73 wall=601.58ms, difference simplify=120.91ms avg_simplify_ms=2.42 wall=396.15ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=969.30ms avg_case_ms=9.69 avg_simplify_ms=2.71, sum@0+100 failed=0 elapsed=670.51ms avg_case_ms=6.71 avg_simplify_ms=2.09, product@0+100 failed=0 elapsed=601.58ms avg_case_ms=6.02 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=396.15ms avg_case_ms=7.92 avg_simplify_ms=2.42, sum@700+100 failed=0 elapsed=231.08ms avg_case_ms=2.31 avg_simplify_ms=0.83
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.57ms median_wire=15.62ms median_wall=58.30ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.52ms median_wire=16.60ms median_wall=63.97ms, product@0+100 #175 product runs=3 median_simplify=15.12ms median_wire=15.16ms median_wall=57.69ms, difference@0+50 #174 difference runs=3 median_simplify=15.69ms median_wire=15.75ms median_wall=58.21ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.93ms median_wire=13.00ms median_wall=49.06ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.87s | passed=450 failed=0 total=450 avg_case=6.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.59s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
