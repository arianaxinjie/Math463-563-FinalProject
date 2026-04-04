# Final Grid Search Report

## Experiment Setup

- Test image: `cameraman`
- Blur: Gaussian, `kernel_size=15`, `kernel_sigma=3.0`
- Noise: salt-and-pepper, `noise_density=0.05`
- Boundary mode: `periodic`
- Iteration budget: `500`
- Problems searched: `l1`, `l2`
- Algorithms searched:
  - `primal_dr`
  - `primal_dual_dr`
  - `admm`
  - `chambolle_pock`

## Search Procedure

Two rounds were run:

1. Stage 1 coarse grid
   - `5412` unique trials
   - covered all four algorithms on both `l1` and `l2`
2. Stage 2 local refinement
   - `stage2_l1`: `910` trials
   - `stage2_l2`: `1136` trials
   - refined neighborhoods around the stage-1 best settings

## Stage 1 Best Results

| Algorithm | Problem | PSNR (dB) | Best Parameters |
| --- | --- | ---: | --- |
| Primal DR | `l1` | `26.2156` | `gamma=0.002`, `tprimaldr=1.0`, `rhoprimaldr=1.5` |
| Primal-Dual DR | `l1` | `27.2397` | `gamma=0.002`, `tprimaldualdr=4.0`, `rhoprimaldualdr=1.5` |
| ADMM | `l1` | `26.3391` | `gamma=0.003`, `tadmm=3.0`, `rhoadmm=1.5` |
| Chambolle-Pock | `l1` | `25.1234` | `gamma=0.003`, `cp_theta=0.9`, `cp_ratio=2.0` |
| Primal DR | `l2` | `21.5789` | `gamma=0.04`, `tprimaldr=1.0`, `rhoprimaldr=0.5` |
| Primal-Dual DR | `l2` | `21.5807` | `gamma=0.04`, `tprimaldualdr=4.0`, `rhoprimaldualdr=0.5` |
| ADMM | `l2` | `21.5782` | `gamma=0.04`, `tadmm=3.0`, `rhoadmm=1.5` |
| Chambolle-Pock | `l2` | `21.5758` | `gamma=0.04`, `cp_theta=0.3`, `cp_ratio=0.5` |

## Stage 2 Best Results

### `l1` refinement

| Algorithm | PSNR (dB) | Best Parameters |
| --- | ---: | --- |
| Primal DR | `26.8231` | `gamma=0.001`, `tprimaldr=1.0`, `rhoprimaldr=1.75` |
| Primal-Dual DR | `27.7613` | `gamma=0.0015`, `tprimaldualdr=5.0`, `rhoprimaldualdr=2.0` |
| ADMM | `26.3374` | `gamma=0.0025`, `tadmm=2.5`, `rhoadmm=1.5` |
| Chambolle-Pock | `25.2397` | `gamma=0.0035`, `cp_theta=0.99`, `cp_ratio=2.5` |

### `l2` refinement

| Algorithm | PSNR (dB) | Best Parameters |
| --- | ---: | --- |
| Primal DR | `21.4210` | `gamma=0.045`, `tprimaldr=1.0`, `rhoprimaldr=0.25` |
| Primal-Dual DR | `21.4367` | `gamma=0.045`, `tprimaldualdr=6.0`, `rhoprimaldualdr=0.35` |
| ADMM | `21.4154` | `gamma=0.045`, `tadmm=5.0`, `rhoadmm=1.25` |
| Chambolle-Pock | `21.4105` | `gamma=0.045`, `cp_theta=0.35`, `cp_ratio=1.0` |

## Comparison and Interpretation

### `l1`

- Stage 2 improved `primal_dr` from `26.2156` to `26.8231` (`+0.6075 dB`)
- Stage 2 improved `primal_dual_dr` from `27.2397` to `27.7613` (`+0.5216 dB`)
- Stage 2 improved `chambolle_pock` from `25.1234` to `25.2397` (`+0.1163 dB`)
- `admm` stayed essentially unchanged

Conclusion for `l1`:
- the refinement helped
- the best final `l1` method is `primal_dual_dr`

### `l2`

- The local refinement did **not** improve the stage-1 coarse-grid best scores
- All stage-2 `l2` best PSNR values were below the stage-1 `l2` best values

Conclusion for `l2`:
- keep the **stage-1** best settings as the final recommendation
- the stage-2 local grid around `gamma≈0.045` was not better than the original coarse optimum near `gamma=0.04`

## Final Recommended Parameters

### Recommended `l1` settings

| Algorithm | Final PSNR (dB) | Final Parameters |
| --- | ---: | --- |
| Primal DR | `26.8231` | `gamma=0.001`, `tprimaldr=1.0`, `rhoprimaldr=1.75` |
| Primal-Dual DR | `27.7613` | `gamma=0.0015`, `tprimaldualdr=5.0`, `rhoprimaldualdr=2.0` |
| ADMM | `26.3391` | `gamma=0.003`, `tadmm=3.0`, `rhoadmm=1.5` |
| Chambolle-Pock | `25.2397` | `gamma=0.0035`, `cp_theta=0.99`, `cp_ratio=2.5` |

### Recommended `l2` settings

| Algorithm | Final PSNR (dB) | Final Parameters |
| --- | ---: | --- |
| Primal DR | `21.5789` | `gamma=0.04`, `tprimaldr=1.0`, `rhoprimaldr=0.5` |
| Primal-Dual DR | `21.5807` | `gamma=0.04`, `tprimaldualdr=4.0`, `rhoprimaldualdr=0.5` |
| ADMM | `21.5782` | `gamma=0.04`, `tadmm=3.0`, `rhoadmm=1.5` |
| Chambolle-Pock | `21.5758` | `gamma=0.04`, `cp_theta=0.3`, `cp_ratio=0.5` |

## Overall Takeaway

- For this `cameraman + Gaussian blur + salt-and-pepper noise` experiment, `l1` fidelity is clearly better than `l2`
- The strongest final method is:
  - `primal_dual_dr` with `l1`
  - PSNR `27.7613 dB`
  - parameters: `gamma=0.0015`, `tprimaldualdr=5.0`, `rhoprimaldualdr=2.0`
