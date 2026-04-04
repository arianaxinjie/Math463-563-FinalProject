# Grid Search Report: Stage 1

## Setup

- Image: `cameraman`
- Blur: Gaussian, `kernel_size=15`, `kernel_sigma=3.0`
- Noise: salt-and-pepper, `noise_density=0.05`
- Boundary mode: `periodic`
- Solvers: `primal_dr`, `primal_dual_dr`, `admm`, `chambolle_pock`
- Fidelity terms searched: `l1`, `l2`
- Iteration budget per trial: `500`
- Total unique trials completed: `5412`

## Main Conclusions

1. The full `l1 + l2` grid search completed successfully.
2. Under salt-and-pepper noise, the best overall result came from `primal_dual_dr` with `l1` fidelity.
3. All four algorithms produced very similar best `l2` scores, clustered around `21.58 dB`.
4. The best `l1` scores were clearly better than the best `l2` scores on this impulse-noise setting, which is consistent with `l1` being a better fidelity model for salt-and-pepper corruption.

## Best Result Per Algorithm and Problem

| Algorithm | Problem | Best PSNR (dB) | Best Parameters |
| --- | --- | ---: | --- |
| Primal DR | `l1` | `26.2156` | `gamma=0.002`, `tprimaldr=1.0`, `rhoprimaldr=1.5` |
| Primal DR | `l2` | `21.5789` | `gamma=0.04`, `tprimaldr=1.0`, `rhoprimaldr=0.5` |
| Primal-Dual DR | `l1` | `27.2397` | `gamma=0.002`, `tprimaldualdr=4.0`, `rhoprimaldualdr=1.5` |
| Primal-Dual DR | `l2` | `21.5807` | `gamma=0.04`, `tprimaldualdr=4.0`, `rhoprimaldualdr=0.5` |
| ADMM | `l1` | `26.3391` | `gamma=0.003`, `tadmm=3.0`, `rhoadmm=1.5` |
| ADMM | `l2` | `21.5782` | `gamma=0.04`, `tadmm=3.0`, `rhoadmm=1.5` |
| Chambolle-Pock | `l1` | `25.1234` | `gamma=0.003`, `cp_theta=0.9`, `cp_ratio=2.0` |
| Chambolle-Pock | `l2` | `21.5758` | `gamma=0.04`, `cp_theta=0.3`, `cp_ratio=0.5` |

## Best Overall by Problem

- `l1`: `primal_dual_dr`, PSNR `27.2397 dB`
- `l2`: `primal_dual_dr`, PSNR `21.5807 dB`

## Interpretation

- For this fixed test setting, `l1` fidelity is the stronger choice.
- `primal_dual_dr` is the strongest candidate to refine first.
- Several winning parameters sit near the edge of the coarse grid:
  - `l1`: small `gamma` near `0.002` and larger step/relaxation values
  - `l2`: `gamma` near `0.04`
- Because of those edge hits, a second-stage local refinement around the current optima is justified.

## Stage 2 Plan

- Run a finer local search around the current best point for each `(algorithm, problem)` pair.
- Split the refinement into two runs:
  - `stage2_l1`
  - `stage2_l2`
- Total planned stage-2 trials:
  - `stage2_l1`: `910`
  - `stage2_l2`: `1136`
  - combined: `2046`
