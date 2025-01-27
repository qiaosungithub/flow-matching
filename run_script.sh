conda activate resnet_jax

# clean tmp
rm -rf tmp

# python3 script.py

PWD=$(pwd)
python3 main.py \
    --debug=False \
    --workdir=${PWD}/tmp --config=configs/tpu.py \
    --config.dataset.cache=True \
    --config.dataset.root=MNIST \
    --config.model.image_size=28 \
    --config.model.out_channels=1 \
    --config.batch_size=32 \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.learning_rate=0.001 \
    --config.lr_schedule=const \
    --config.weight_decay=0 \
    --config.optimizer=adamw \
    --config.adam_b2=0.999 \
    --config.warmup_epochs=200 \
    --config.log_per_step=100 \
    --config.eval_per_epoch=1 \
    --config.visualize_per_epoch=2 \
    --config.num_epochs=4000 \
    --config.model.base_width=32 \
    --config.model.n_T=32 \
    --config.fid.on_use=False \
    --config.fid.eval_only=False \
    --config.fid.fid_per_epoch=1 \
    --config.fid.num_samples=5000 \
    --config.fid.device_batch_size=128 \
    --config.model.net_type='context' \
    --config.model.dropout=0.2 \
    --config.aug.use_edm_aug=True \
    --config.model.use_aug_label=True \
    --config.model.t_conditioned=True \
    # --config.restore=/kmh-nfs-ssd-eu-mount/logs/kaiminghe/hvae/iddpm/20241230_191454_6b0964_kmh-tpuvm-v3-32-5_tpu_b512_constlr0.0004_wd0_adam_ep4000wm200_w128_n18_dp0.2_uncond_fid_ncsnppedm_auglabel_0tcond_iddpm_b0.95_edmsampler
    # --config.restore=/kmh-nfs-ssd-eu-mount/logs/kaiminghe/hvae/dbg/20241229_043447_s8mrhy_kmh-tpuvm-v4-8-8_tpu_b512_constlr0.0004_wd0_adam_ep4000wm200_w128_n128_dp0.2_uncond_fid_ncsnppedm_auglabel_1tcond_iddpm_b0.95_rangeM/checkpoint_135800
    # --config.restore=/kmh-nfs-ssd-eu-mount/logs/kaiminghe/hvae/dbg/20241229_022410_bf58l4_kmh-tpuvm-v3-32-5_tpu_b512_constlr0.0004_wd0_adam_ep4000wm200_w128_n128_dp0.2_uncond_fid_ncsnppedm_NOauglabel_1tcond_iddpm_b0.95_rangeM/checkpoint_58200    

    # --config.model.net_type='context' \
    # --config.model.net_type='ncsnpp' \
    # --config.model.net_type='ncsnppedm' \
    # --config.restore=/kmh-nfs-ssd-eu-mount/logs/kaiminghe/hvae/edm/20240820_023513_643o22_kmh-tpuvm-v4-8-7_tpu_b512_constlr0.001_wd0_adam_ep4000wm200_w128_uncond





