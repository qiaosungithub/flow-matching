It would be nice to include DPM-Solver, DPM-Solver++-type samplers as well, since these are still well-used samplers in practice, and also these samplers use multiple tabbed filtering. The performance for these type of samplers might be closely related to DDIM's failure as well.

DDIM ckpt: `/kmh-nfs-us-mount/logs/sqa/sqa_Flow_matching/20241209_004559_gihhxu_kmh-tpuvm-v2-32-1__b_lr_ep_eval/checkpoint_194000` (1000 linear)

# DPM-solver

| Sampler | NFE | FID-50k |
| --- | --- | --- |
| DDIM | 100 | 3.99 (baseline) |