# EE-ARF-10 test-split summary

Seeds: [42, 7, 123] (metrics are seed means). Config SHA-256: `053e5700259c8aeca58d95b217ef626e4ecea2e2fd1c64a41cdb740cd93a43d0`

## Spain (ENTSO-E)

| Model | Acc % | κ_temporal (±sd) | κ_Cohen | Macro-F1 | PR-AUC | FNR | ρ % | Trees eval | Trees trained | Mean µs | P99 ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always-HT | 85.06 | 0.2155 (±0.0000) | 0.6968 | 0.8484 | 0.8311 | 0.1801 | 0.0 | 1.00 | 1.00 | 4.5 | 0.011 | 0.089 |
| Always-HAT | 84.71 | 0.1973 (±0.0208) | 0.6897 | 0.8448 | 0.8526 | 0.1848 | 0.0 | 1.00 | 1.00 | 84.2 | 0.211 | 1.635 |
| Always-ARF-2 | 83.70 | 0.1442 (±0.0999) | 0.6688 | 0.8344 | 0.8958 | 0.2005 | 0.0 | 2.00 | 2.00 | 21.0 | 0.039 | 0.242 |
| Always-ARF-3 | 85.97 | 0.2636 (±0.0479) | 0.7145 | 0.8572 | 0.9200 | 0.1842 | 0.0 | 3.00 | 3.00 | 23.9 | 0.042 | 1.086 |
| Always-ARF-10 | 89.20 | 0.4331 (±0.0276) | 0.7804 | 0.8901 | 0.9501 | 0.1455 | 100.0 | 10.00 | 10.00 | 64.1 | 0.118 | 0.557 |
| EE-ARF-10 (Margin-Only Static Gate) | 88.46 | 0.3944 (±0.0246) | 0.7655 | 0.8827 | 0.9333 | 0.1514 | 27.6 | 4.93 | 5.95 | 40.5 | 0.092 | 0.973 |
| EE-ARF-10 (Dormant Replay B=200) | 88.14 | 0.3773 (±0.0015) | 0.7588 | 0.8794 | 0.9312 | 0.1550 | 27.1 | 4.90 | 5.01 | 35.4 | 0.087 | 5.969 |
| EE-ARF-10 (Conformal, final-error loss) | 89.23 | 0.4346 (±0.0043) | 0.7811 | 0.8905 | 0.9491 | 0.1422 | 98.4 | 9.89 | 9.91 | 64.6 | 0.119 | 0.604 |
| **EE-ARF-10 (Conformal Gate + Subsampled BG)** | 88.62 | 0.4023 (±0.0075) | 0.7686 | 0.8843 | 0.9348 | 0.1475 | 26.1 | 4.82 | 5.87 | 37.9 | 0.088 | 2.212 |

Dominance (proposed vs. baseline):

- vs Always-ARF-3: Δκ_T = +0.1388; latency ×1.58; trees ×1.61 → (κ_T, latency): **trade-off**, (κ_T, trees): **trade-off**
- vs Always-ARF-10: Δκ_T = -0.0308; latency ×0.59; trees ×0.48 → (κ_T, latency): **trade-off**, (κ_T, trees): **trade-off**
- Gap closed from ARF-3 to ARF-10 (κ_T): 81.9%

## India (POSOCO/NLDC)

| Model | Acc % | κ_temporal (±sd) | κ_Cohen | Macro-F1 | PR-AUC | FNR | ρ % | Trees eval | Trees trained | Mean µs | P99 ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always-HT | 81.32 | -0.2032 (±0.0000) | 0.5232 | 0.7598 | 0.7240 | 0.4346 | 0.0 | 1.00 | 1.00 | 4.3 | 0.010 | 0.070 |
| Always-HAT | 84.24 | -0.0151 (±0.0689) | 0.6145 | 0.8069 | 0.7615 | 0.3161 | 0.0 | 1.00 | 1.00 | 86.1 | 0.269 | 3.202 |
| Always-ARF-2 | 85.63 | 0.0746 (±0.0422) | 0.6470 | 0.8232 | 0.8360 | 0.2995 | 0.0 | 2.00 | 2.00 | 21.1 | 0.041 | 0.805 |
| Always-ARF-3 | 87.02 | 0.1637 (±0.0295) | 0.6745 | 0.8365 | 0.8626 | 0.3109 | 0.0 | 3.00 | 3.00 | 23.5 | 0.042 | 0.273 |
| Always-ARF-10 | 88.45 | 0.2560 (±0.0227) | 0.7088 | 0.8536 | 0.9034 | 0.2966 | 100.0 | 10.00 | 10.00 | 64.0 | 0.129 | 1.994 |
| EE-ARF-10 (Margin-Only Static Gate) | 88.89 | 0.2845 (±0.0061) | 0.7239 | 0.8616 | 0.8814 | 0.2651 | 31.6 | 5.21 | 5.21 | 38.0 | 0.096 | 0.432 |
| EE-ARF-10 (Dormant Replay B=200) | 88.55 | 0.2621 (±0.0062) | 0.7133 | 0.8561 | 0.8844 | 0.2825 | 39.6 | 5.78 | 6.14 | 41.3 | 0.099 | 6.116 |
| EE-ARF-10 (Conformal, final-error loss) | 88.83 | 0.2807 (±0.0137) | 0.7206 | 0.8597 | 0.9026 | 0.2773 | 90.8 | 9.35 | 9.35 | 60.1 | 0.122 | 0.407 |
| **EE-ARF-10 (Conformal Gate + Subsampled BG)** | 89.02 | 0.2926 (±0.0085) | 0.7256 | 0.8623 | 0.8899 | 0.2721 | 38.6 | 5.70 | 5.70 | 38.8 | 0.097 | 0.318 |

Dominance (proposed vs. baseline):

- vs Always-ARF-3: Δκ_T = +0.1288; latency ×1.65; trees ×1.90 → (κ_T, latency): **trade-off**, (κ_T, trees): **trade-off**
- vs Always-ARF-10: Δκ_T = +0.0366; latency ×0.61; trees ×0.57 → (κ_T, latency): **dominates**, (κ_T, trees): **dominates**
- Gap closed from ARF-3 to ARF-10 (κ_T): 139.6%

Notes: latency = one predict+learn step on this host (Python/river), not an embedded target. Avg trees evaluated ≥ 3 for EE-ARF-10 by construction (always-on core).
