# Engine Improvement Scorecard

- Generated: 2026-06-02T04:10:33.585063+00:00
- Git branch: main
- Git commit: `4bce7b2daa0376a1e400e796430bec1a39614461`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=2
- By area: calculus / differentiation:1, calculus / domain-condition display:1
- Recent 1: `calculus / domain-condition display` - 2026-05-28 - Observe-only discovery: condition display must not scale-normalize periodic arguments
- Recent 2: `calculus / differentiation` - 2026-05-28 - Discovery observe-only: atanh exact-square denominator scale hits depth overflow

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=263
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=340

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=611.52ms avg_case_ms=6.12 simplify=31.30ms avg_simplify_ms=0.31, sum total=200 failed=0 elapsed=509.87ms avg_case_ms=2.55 simplify=74.13ms avg_simplify_ms=0.37, product total=100 failed=0 elapsed=340.51ms avg_case_ms=3.41 simplify=20.43ms avg_simplify_ms=0.20, difference total=50 failed=0 elapsed=240.83ms avg_case_ms=4.82 simplify=27.42ms avg_simplify_ms=0.55
- Engine hotspots: sum simplify=74.13ms avg_simplify_ms=0.37 wall=509.87ms, shifted_quotient simplify=31.30ms avg_simplify_ms=0.31 wall=611.52ms, difference simplify=27.42ms avg_simplify_ms=0.55 wall=240.83ms, product simplify=20.43ms avg_simplify_ms=0.20 wall=340.51ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=611.52ms avg_case_ms=6.12 avg_simplify_ms=0.31, sum@0+100 failed=0 elapsed=363.28ms avg_case_ms=3.63 avg_simplify_ms=0.43, product@0+100 failed=0 elapsed=340.51ms avg_case_ms=3.41 avg_simplify_ms=0.20, difference@0+50 failed=0 elapsed=240.83ms avg_case_ms=4.82 avg_simplify_ms=0.55, sum@700+100 failed=0 elapsed=146.59ms avg_case_ms=1.47 avg_simplify_ms=0.31
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.26ms median_wire=5.30ms median_wall=5.61ms, sum@0+100 #53 sum runs=3 median_simplify=3.90ms median_wire=3.94ms median_wall=5.37ms, difference@0+50 #54 difference runs=3 median_simplify=3.58ms median_wire=3.62ms median_wall=5.04ms, sum@0+100 #209 sum runs=3 median_simplify=3.27ms median_wire=3.32ms median_wall=5.41ms, sum@700+100 #3021 sum runs=3 median_simplify=3.41ms median_wire=3.44ms median_wall=3.73ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), difference@0+50 #54 difference expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.70s | passed=450 failed=0 total=450 avg_case=3.778ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.54s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.12s | passed=1 failed=0 |
