It would be nice to include DPM-Solver, DPM-Solver++-type samplers as well, since these are still well-used samplers in practice, and also these samplers use multiple tabbed filtering. The performance for these type of samplers might be closely related to DDIM's failure as well.

DDIM ckpt: `/kmh-nfs-us-mount/logs/sqa/sqa_Flow_matching/20241209_004559_gihhxu_kmh-tpuvm-v2-32-1__b_lr_ep_eval/checkpoint_194000` (1000 linear)

(Copied from the authors of DPM:)

Some advices for choosing the algorithm:
    - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
        Use singlestep DPM-Solver or DPM-Solver++ ("DPM-Solver-fast" in the paper) with `order = 3`.
        e.g., DPM-Solver:
            >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
            >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                    skip_type='time_uniform', method='singlestep')
        e.g., DPM-Solver++:
            >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
            >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                    skip_type='time_uniform', method='singlestep')
    - For **guided sampling with large guidance scale** by DPMs:
        Use multistep DPM-Solver with `algorithm_type="dpmsolver++"` and `order = 2`.
        e.g.
            >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
            >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                    skip_type='time_uniform', method='multistep')

We support three types of `skip_type`:
    - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
    - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
    - 'time_quadratic': quadratic time for the time steps.

# DPM-solver

| Sampler | NFE | Model | FID-50k |
| --- | --- | --- | --- |
| DDIM | 100 | Normal | 3.99 (baseline) |
| DDIM | 100 | w/o t | 49.29 (baseline) |
| DPM-Solver | 10 | Normal | 4.68 |
| DPM-Solver | 20 | Normal | 3.94 |
| DPM-Solver++ | 10 | Normal | 5.66 |
| DPM-Solver++ | 20 | Normal | 3.84 |
| DPM-Solver | 10 | w/o t | 68.81 |
| DPM-Solver | 20 | w/o t | 62.44 |
| DPM-Solver++ | 10 | w/o t | 78.47 |
| DPM-Solver++ | 20 | w/o t | 54.81 |