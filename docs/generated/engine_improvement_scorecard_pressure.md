# Engine Improvement Scorecard

- Generated: 2026-07-08T08:48:27.913205+00:00
- Git branch: main
- Git commit: `ffd17708df921eb4ccd640ff6685045e6b6af8bd`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=972.03ms avg_case_ms=9.72 simplify=269.71ms avg_simplify_ms=2.70, sum total=200 failed=0 elapsed=835.37ms avg_case_ms=4.18 simplify=268.32ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=592.70ms avg_case_ms=5.93 simplify=168.73ms avg_simplify_ms=1.69, difference total=50 failed=0 elapsed=391.08ms avg_case_ms=7.82 simplify=117.87ms avg_simplify_ms=2.36
- Engine hotspots: shifted_quotient simplify=269.71ms avg_simplify_ms=2.70 wall=972.03ms, sum simplify=268.32ms avg_simplify_ms=1.34 wall=835.37ms, product simplify=168.73ms avg_simplify_ms=1.69 wall=592.70ms, difference simplify=117.87ms avg_simplify_ms=2.36 wall=391.08ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=972.03ms avg_case_ms=9.72 avg_simplify_ms=2.70, sum@0+100 failed=0 elapsed=607.20ms avg_case_ms=6.07 avg_simplify_ms=1.87, product@0+100 failed=0 elapsed=592.70ms avg_case_ms=5.93 avg_simplify_ms=1.69, difference@0+50 failed=0 elapsed=391.08ms avg_case_ms=7.82 avg_simplify_ms=2.36, sum@700+100 failed=0 elapsed=228.17ms avg_case_ms=2.28 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.71ms median_wire=16.77ms median_wall=65.38ms, product@0+100 #175 product runs=3 median_simplify=16.32ms median_wire=16.38ms median_wall=59.97ms, difference@0+50 #174 difference runs=3 median_simplify=14.81ms median_wire=14.86ms median_wall=56.66ms, sum@0+100 #173 sum runs=3 median_simplify=14.69ms median_wire=14.74ms median_wall=56.58ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.66ms median_wire=12.72ms median_wall=48.35ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.79s | passed=450 failed=0 total=450 avg_case=6.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.08s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
